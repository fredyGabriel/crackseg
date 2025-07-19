"""Model initialization and channel detection for Swin Transformer V2 Encoder.

This module handles the complex initialization process of Swin Transformer
models, including model loading, channel dimension detection, and validation.
"""

import logging
from typing import TYPE_CHECKING

import timm
import torch

if TYPE_CHECKING:
    from crackseg.model.encoder.swin.config import SwinTransformerEncoderConfig

logger = logging.getLogger(__name__)


class SwinModelInitializer:
    """
    Handles initialization and channel detection for Swin Transformer models.
    """

    @staticmethod
    def initialize_swin_model(
        in_channels: int,
        config: "SwinTransformerEncoderConfig",
        out_indices: list[int],
    ) -> torch.nn.Module:
        """Initialize Swin Transformer model with robust error handling.

        Attempts to create the specified Swin Transformer model from the timm
        library with automatic fallback to ResNet34 if initialization fails.
        Provides comprehensive logging for debugging initialization issues.

        Args:
            in_channels: Number of input channels for model creation.
            config: Configuration object containing model specifications.
            out_indices: Which stages to extract features from.

        Returns:
            Initialized model instance.

        Raises:
            ValueError: If both primary and fallback model initialization fail.

        Error Handling:
            - Primary model initialization with comprehensive exception
            catching
            - Automatic fallback to ResNet34 encoder if Swin model fails
            - Detailed logging of initialization attempts and failures
            - Graceful error recovery with informative error messages

        Model Creation Parameters:
            - model_name: Swin variant identifier from config
            - pretrained: Whether to load ImageNet pretrained weights
            - in_chans: Input channel count for first conv layer
            - features_only: Extract features without classification head
            - out_indices: Which stages to extract features from

        Notes:
            - Internet connection required for pretrained weight download
            - First-time model download may take several minutes
            - Fallback model provides basic functionality if Swin fails
        """
        try:
            logger.info(
                f"Attempting to initialize {config.model_name} with "
                f"img_size={config.img_size}"
            )
            # timm.create_model already returns nn.Module, no cast needed
            model = timm.create_model(
                config.model_name,
                pretrained=config.pretrained,
                in_chans=in_channels,
                features_only=config.features_only,
                out_indices=out_indices,
            )
            logger.info(f"Successfully initialized {config.model_name} model")
            return model
        except (RuntimeError, ValueError, TypeError, FileNotFoundError) as e:
            logger.error(f"Failed to initialize {config.model_name}: {str(e)}")
            logger.error("Falling back to ResNet-based encoder")
            try:
                fallback_model = "resnet34"
                # timm.create_model already returns nn.Module, no cast needed
                model = timm.create_model(
                    fallback_model,
                    pretrained=config.pretrained,
                    in_chans=in_channels,
                    features_only=True,
                    out_indices=out_indices,
                )
                logger.info(
                    f"Successfully initialized fallback model {fallback_model}"
                )
                return model
            except (
                RuntimeError,
                ValueError,
                TypeError,
                FileNotFoundError,
            ) as e2:
                logger.error(f"Failed to initialize fallback model: {str(e2)}")
                raise ValueError(
                    f"Could not initialize any model: {str(e2)}"
                ) from e2

    @staticmethod
    def determine_channel_info(
        model: torch.nn.Module, in_channels: int, img_size: int
    ) -> tuple[list[int], int]:
        """Determine skip_channels and out_channels using a dummy pass or
        feature_info.

        Args:
            model: The initialized model instance.
            in_channels: Number of input channels.
            img_size: Expected input image size.

        Returns:
            Tuple of (skip_channels, out_channels).

        Raises:
            RuntimeError: If channel detection fails completely.
        """
        try:
            dummy_input = torch.randn(2, in_channels, img_size, img_size)
            model.eval()
            with torch.no_grad():
                dummy_output = model(dummy_input)
            # Handle different return types from timm models
            if isinstance(dummy_output, torch.Tensor):
                dummy_features: list[torch.Tensor] = [dummy_output]
            elif isinstance(dummy_output, list | tuple):
                dummy_features = [
                    feat
                    for feat in dummy_output
                    if isinstance(feat, torch.Tensor)
                ]
            else:
                raise TypeError(
                    f"Unexpected output type: {type(dummy_output)}"
                )

            actual_all_channels = [feat.shape[1] for feat in dummy_features]
            logger.info(
                "Detected feature channels from dummy pass: "
                f"{actual_all_channels}"
            )
            skip_channels = list(reversed(actual_all_channels[:-1]))
            out_channels = actual_all_channels[-1]
            return skip_channels, out_channels
        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            logger.error(
                f"Failed to determine output channels via dummy forward pass: "
                f"{e}. Falling back to timm feature_info."
            )
            # Handle feature_info access more carefully with type checking
            feature_info = getattr(model, "feature_info", None)
            if feature_info is not None:
                info_list = getattr(feature_info, "info", None)
                if info_list is not None:
                    # Extract channel information from feature_info
                    all_channels: list[int] = []
                    for info in info_list:
                        channels: int | None = None
                        if isinstance(info, dict) and "num_chs" in info:
                            channels = info["num_chs"]
                        else:
                            # Try to access as attribute
                            channels = getattr(info, "num_chs", None)

                        if isinstance(channels, int):
                            all_channels.append(channels)

                    if all_channels:
                        skip_channels = list(reversed(all_channels[:-1]))
                        out_channels = all_channels[-1]
                        logger.warning(
                            f"Using fallback channels from feature_info: "
                            f"skips={skip_channels}, "
                            f"out={out_channels}"
                        )
                        return skip_channels, out_channels
                    else:
                        raise RuntimeError(
                            "Could not extract channel information "
                            "from feature_info."
                        ) from e
                else:
                    raise RuntimeError(
                        "feature_info.info not available or empty."
                    ) from e
            else:
                logger.error(
                    "Cannot determine output channels from dummy pass or "
                    "feature_info."
                )
                raise RuntimeError(
                    "Could not determine encoder output channel dimensions."
                ) from e

    @staticmethod
    def calculate_reduction_factors(
        model: torch.nn.Module, skip_channels: list[int], patch_size: int
    ) -> list[int]:
        """Calculate and store reduction factors for each feature stage.

        Args:
            model: The initialized model instance.
            skip_channels: List of skip connection channel counts.
            patch_size: Size of image patches.

        Returns:
            List of reduction factors for each stage.
        """
        feature_info = getattr(model, "feature_info", None)
        if feature_info is not None:
            info_list = getattr(feature_info, "info", None)
            if info_list is not None:
                reduction_factors: list[int] = []
                for i, info in enumerate(info_list):
                    reduction: int = 2 ** (i + 1)  # Default fallback
                    if isinstance(info, dict):
                        reduction = info.get("reduction", reduction)
                    else:
                        # Try to access as attribute
                        reduction = getattr(info, "reduction", reduction)
                    reduction_factors.append(reduction)
                return reduction_factors
            else:
                num_stages = len(skip_channels) + 1
                reduction_factors = [
                    patch_size * (2**i) for i in range(num_stages)
                ]
                logger.warning(
                    f"Estimating reduction factors: {reduction_factors}"
                )
                return reduction_factors
        else:
            num_stages = len(skip_channels) + 1
            reduction_factors = [
                patch_size * (2**i) for i in range(num_stages)
            ]
            logger.warning(
                f"Estimating reduction factors: {reduction_factors}"
            )
            return reduction_factors

    @staticmethod
    def validate_model_config(
        model_name: str, img_size: int, min_model_name_parts: int
    ) -> None:
        """Validate that the model name is compatible with the configured
        image size.

        Args:
            model_name: The model name to validate.
            img_size: The configured image size.
            min_model_name_parts: Minimum number of parts for size checking.
        """
        # Extract the image size from the model name if present
        if "_" in model_name:
            name_parts = model_name.split("_")
            if len(name_parts) >= min_model_name_parts:
                # Last part might contain the image size (e.g., "256")
                try:
                    name_img_size = int(name_parts[-1])
                    if name_img_size != img_size:
                        logger.warning(
                            f"Model {model_name} is designed for "
                            f"{name_img_size}x{name_img_size} images, "
                            f"but img_size={img_size} was specified."
                        )
                except ValueError:
                    # Not a number, so can't determine image size from name
                    pass
