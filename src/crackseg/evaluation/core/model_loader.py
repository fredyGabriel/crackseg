"""Model loading utilities for crack segmentation evaluation."""

import importlib
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of trained models and configurations."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        """
        Initialize the model loader.

        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint with PyTorch 2.7+ compatibility."""
        try:
            # Try with weights_only=False for full checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load full checkpoint: {e}")
            # Fallback to weights_only=True
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )

        return checkpoint

    def load_config(
        self, config_path: str | Path | None = None
    ) -> DictConfig | ListConfig:
        """
        Load model configuration.

        Args:
            config_path: Optional path to config file

        Returns:
            Model configuration
        """
        checkpoint = self.load_checkpoint()

        # Try config_path first
        if config_path and Path(config_path).exists():
            config = OmegaConf.load(config_path)
        elif "config" in checkpoint:
            config = OmegaConf.create(checkpoint["config"])
        else:
            raise ValueError(
                "No configuration found in checkpoint or config_path"
            )

        return config

    def create_model(self, config: DictConfig) -> torch.nn.Module:
        """
        Create model from configuration.

        Args:
            config: Model configuration

        Returns:
            Instantiated PyTorch model
        """
        # Load checkpoint first to analyze architecture
        checkpoint = self.load_checkpoint()

        # Detect actual architecture from checkpoint
        detected_config = self._detect_architecture_from_checkpoint(
            checkpoint, config
        )
        model_config = detected_config.model

        if not hasattr(model_config, "_target_"):
            raise ValueError("Model configuration must have _target_ field")

        # Import model class dynamically
        module_path, class_name = model_config._target_.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Check if this is a hybrid architecture that needs component instantiation
        if self._is_hybrid_architecture(model_config):
            model = self._create_hybrid_model(model_class, model_config)
        else:
            # Filter out _target_ from config for simple models
            model_kwargs = {
                k: v for k, v in model_config.items() if k != "_target_"
            }
            model = model_class(**model_kwargs)

        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded model weights from epoch "
                f"{checkpoint.get('epoch', 'unknown')}"
            )
        else:
            raise ValueError("No model_state_dict found in checkpoint")

        return model

    def _is_hybrid_architecture(self, model_config: DictConfig) -> bool:
        """Check if the model configuration represents a hybrid architecture."""
        hybrid_indicators = ["encoder", "bottleneck", "decoder"]
        return any(
            hasattr(model_config, component) for component in hybrid_indicators
        )

    def _model_uses_cfg_suffix(self, model_class: type) -> bool:
        """Check if the model class uses 'cfg' suffix for component parameters."""
        import inspect

        sig = inspect.signature(model_class.__init__)
        params = list(sig.parameters.keys())
        return any("_cfg" in param for param in params)

    def _detect_architecture_from_checkpoint(
        self, checkpoint: dict[str, Any], fallback_config: DictConfig
    ) -> DictConfig:
        """
        Detect the actual model architecture from checkpoint state_dict.

        Args:
            checkpoint: Loaded checkpoint dictionary
            fallback_config: Fallback configuration if detection fails

        Returns:
            Configuration with corrected model architecture
        """
        if "model_state_dict" not in checkpoint:
            logger.warning(
                "No model_state_dict in checkpoint, using fallback config"
            )
            return fallback_config

        state_dict = checkpoint["model_state_dict"]

        # Analyze state_dict keys to determine architecture
        architecture_info = self._analyze_state_dict_keys(state_dict)

        # If we have original config in checkpoint, use it
        if "config" in checkpoint and "model" in checkpoint["config"]:
            original_model_config = checkpoint["config"]["model"]
            logger.info("Using original model configuration from checkpoint")

            # Create new config with original model config
            new_config = OmegaConf.create(dict(fallback_config))
            new_config.model = OmegaConf.create(original_model_config)
            return new_config

        # Otherwise, try to reconstruct config based on state_dict analysis
        elif architecture_info["is_swinv2_hybrid"]:
            logger.info("Detected SwinV2 hybrid architecture from state_dict")
            return self._create_swinv2_hybrid_config(
                fallback_config, architecture_info
            )

        else:
            logger.warning(
                "Could not detect architecture, using fallback config"
            )
            return fallback_config

    def _analyze_state_dict_keys(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze state_dict keys to determine model architecture."""
        keys = list(state_dict.keys())

        # Detect SwinV2 encoder
        swin_keys = [k for k in keys if "encoder.encoder.swin" in k]

        # Detect ASPP bottleneck
        aspp_keys = [
            k
            for k in keys
            if any(
                pattern in k
                for pattern in [
                    "bottleneck.branches",
                    "bottleneck.conv_1x1",
                    "bottleneck.global_pool",
                ]
            )
        ]

        # Detect CNN decoder with CBAM
        decoder_cbam_keys = [
            k for k in keys if "decoder.decoder_blocks" in k and "cbam" in k
        ]
        decoder_conv_keys = [
            k for k in keys if "decoder.decoder_blocks" in k and "conv" in k
        ]

        return {
            "is_swinv2_hybrid": len(swin_keys) > 0 and len(aspp_keys) > 0,
            "has_swin_encoder": len(swin_keys) > 0,
            "has_aspp_bottleneck": len(aspp_keys) > 0,
            "has_cbam_decoder": len(decoder_cbam_keys) > 0,
            "has_cnn_decoder": len(decoder_conv_keys) > 0,
            "swin_keys_count": len(swin_keys),
            "aspp_keys_count": len(aspp_keys),
            "decoder_keys_count": len(decoder_conv_keys),
        }

    def _create_swinv2_hybrid_config(
        self, base_config: DictConfig, arch_info: dict[str, Any]
    ) -> DictConfig:
        """Create SwinV2 hybrid configuration based on detected architecture."""
        # Create a copy of the base config
        new_config = OmegaConf.create(dict(base_config))

        # Set the correct model target
        new_config.model = OmegaConf.create(
            {
                "_target_": "crackseg.model.architectures.swinv2_cnn_aspp_unet.SwinV2CnnAsppUNet",
                "encoder": {
                    "_target_": "crackseg.model.encoder.swin_v2_adapter.SwinV2EncoderAdapter",
                    "model_name": "swinv2_tiny_window8_256",
                    "img_size": 256,
                    "target_img_size": 256,
                    "in_channels": 3,
                },
                "bottleneck": {
                    "_target_": "crackseg.model.components.aspp.ASPPModule",
                    "in_channels": 768,
                    "output_channels": 256,
                    "dilation_rates": [1, 6, 12, 18],
                    "dropout_rate": 0.1,
                },
                "decoder": {
                    "_target_": "crackseg.model.decoder.cnn_decoder.CNNDecoder",
                    "in_channels": 256,
                    "skip_channels_list": [
                        384,
                        192,
                        96,
                    ],  # SwinV2 skip channels (reversed)
                    "config": {
                        "use_cbam": True,
                        "cbam_reduction": 16,
                        "upsample_mode": "bilinear",
                        "kernel_size": 3,
                        "padding": 1,
                        "upsample_scale_factor": 2,
                    },
                },
                "num_classes": 1,
                "final_activation": "sigmoid",
            }
        )

        logger.info(
            "Created SwinV2CnnAsppUNet configuration from state_dict analysis"
        )
        return new_config

    def _create_hybrid_model(
        self, model_class: type, model_config: DictConfig
    ) -> torch.nn.Module:
        """Create a hybrid model by instantiating components using factory pattern."""
        from crackseg.model.factory.config import (
            instantiate_bottleneck,
            instantiate_decoder,
            instantiate_encoder,
        )

        # Determine parameter naming convention based on model class
        uses_cfg_suffix = self._model_uses_cfg_suffix(model_class)
        encoder_key = "encoder_cfg" if uses_cfg_suffix else "encoder"
        bottleneck_key = "bottleneck_cfg" if uses_cfg_suffix else "bottleneck"
        decoder_key = "decoder_cfg" if uses_cfg_suffix else "decoder"

        # Prepare arguments for the hybrid model
        model_kwargs = {}

        # Handle encoder
        if hasattr(model_config, "encoder"):
            encoder_config = model_config.encoder
            if hasattr(encoder_config, "_target_"):
                # Instantiate encoder directly from config
                encoder_module_path, encoder_class_name = (
                    encoder_config._target_.rsplit(".", 1)
                )
                encoder_module = importlib.import_module(encoder_module_path)
                encoder_class = getattr(encoder_module, encoder_class_name)
                encoder_kwargs = {
                    k: v for k, v in encoder_config.items() if k != "_target_"
                }
                model_kwargs[encoder_key] = encoder_class(**encoder_kwargs)
            else:
                # Use factory pattern - convert DictConfig to dict
                encoder_dict = dict(encoder_config)
                model_kwargs[encoder_key] = instantiate_encoder(encoder_dict)

        # Handle bottleneck
        if hasattr(model_config, "bottleneck"):
            bottleneck_config = model_config.bottleneck
            if hasattr(bottleneck_config, "_target_"):
                # Instantiate bottleneck directly from config
                bottleneck_module_path, bottleneck_class_name = (
                    bottleneck_config._target_.rsplit(".", 1)
                )
                bottleneck_module = importlib.import_module(
                    bottleneck_module_path
                )
                bottleneck_class = getattr(
                    bottleneck_module, bottleneck_class_name
                )
                bottleneck_kwargs = {
                    k: v
                    for k, v in bottleneck_config.items()
                    if k != "_target_"
                }
                model_kwargs[bottleneck_key] = bottleneck_class(
                    **bottleneck_kwargs
                )
            else:
                # Use factory pattern - convert DictConfig to dict
                bottleneck_dict = dict(bottleneck_config)
                model_kwargs[bottleneck_key] = instantiate_bottleneck(
                    bottleneck_dict
                )

        # Handle decoder
        if hasattr(model_config, "decoder"):
            decoder_config = model_config.decoder
            if hasattr(decoder_config, "_target_"):
                # Instantiate decoder directly from config
                decoder_module_path, decoder_class_name = (
                    decoder_config._target_.rsplit(".", 1)
                )
                decoder_module = importlib.import_module(decoder_module_path)
                decoder_class = getattr(decoder_module, decoder_class_name)
                decoder_kwargs = {
                    k: v for k, v in decoder_config.items() if k != "_target_"
                }
                model_kwargs[decoder_key] = decoder_class(**decoder_kwargs)
            else:
                # Use factory pattern - convert DictConfig to dict
                decoder_dict = dict(decoder_config)
                model_kwargs[decoder_key] = instantiate_decoder(decoder_dict)

        # Add other model parameters (excluding component configs)
        excluded_keys = {"_target_", "encoder", "bottleneck", "decoder"}
        for key, value in model_config.items():
            if key not in excluded_keys:
                model_kwargs[key] = value

        logger.info(
            f"Creating hybrid model {model_class.__name__} with components"
        )
        return model_class(**model_kwargs)

    def get_model_info(self, config: DictConfig) -> dict[str, Any]:
        """
        Get information about the model.

        Args:
            config: Model configuration

        Returns:
            Dictionary with model information
        """
        model = self.create_model(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        return {
            "model_type": config.model._target_,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_path": str(self.checkpoint_path),
            "input_size": config.data.image_size,
            "num_classes": config.model.num_classes,
        }
