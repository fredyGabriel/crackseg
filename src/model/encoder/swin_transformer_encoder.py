# a highly cohesive, self-contained SwinTransformerEncoder class. All methods
# are tightly coupled to the encoder logic and splitting would reduce clarity.
# This exception is documented as per the coding-preferences guidelines.

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import timm
import torch
import torch.nn.functional as F

from src.model.base import EncoderBase

logger = logging.getLogger(__name__)


@dataclass
class SwinTransformerEncoderConfig:
    """Configuration for SwinTransformerEncoder."""

    model_name: str = "swinv2_tiny_window16_256"
    pretrained: bool = True
    output_hidden_states: bool = True
    features_only: bool = True
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    img_size: int = 256
    patch_size: int = 4
    use_abs_pos_embed: bool = True
    output_norm: bool = True
    handle_input_size: str = "resize"  # Options: "resize", "pad", "none"
    freeze_layers: bool | str | list[str] = False
    finetune_lr_scale: dict[str, float] | None = None
    min_model_name_parts_for_size_check: int = 3
    expected_input_dims: int = 4


class SwinTransformerEncoder(EncoderBase):
    """
    Swin Transformer V2 Encoder for segmentation tasks.

    This encoder uses the Swin Transformer V2 architecture from the timm
    library and adapts it for use in a U-Net architecture. It extracts
    multi-scale features from the input and returns them in a format
    suitable for a decoder with skip connections.

    Features of Swin Transformer V2:
    - Post-normalization for improved training stability
    - Scaled cosine attention for better performance at high resolutions
    - Log-spaced continuous relative position bias for better transferability
    - Improved architecture with more flexible parameter configurations

    Each stage of the Swin Transformer performs patch merging that reduces the
    resolution by 2x, similar to max pooling in CNNs.
    """

    def _initialize_attributes(
        self, config: SwinTransformerEncoderConfig
    ) -> None:
        """Initialize instance attributes from the configuration."""
        self.out_indices = config.out_indices
        self._skip_channels: list[int] = []  # To be populated
        self._out_channels: int = 0  # To be populated
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.handle_input_size = config.handle_input_size
        self.use_abs_pos_embed = config.use_abs_pos_embed
        self.output_norm = config.output_norm
        self.freeze_layers = config.freeze_layers
        self.finetune_lr_scale: dict[str, float] = (
            config.finetune_lr_scale
            if config.finetune_lr_scale is not None
            else {}
        )
        self.reduction_factors: list[int] = []  # To be populated
        self.min_model_name_parts_for_size_check = (
            config.min_model_name_parts_for_size_check
        )
        self.expected_input_dims = config.expected_input_dims

    def _validate_config(self, config: SwinTransformerEncoderConfig) -> None:
        """Validate specific configuration parameters."""
        self._validate_model_config(config.model_name, self.img_size)
        valid_size_handlers = ["resize", "pad", "none"]
        if self.handle_input_size not in valid_size_handlers:
            raise ValueError(
                f"handle_input_size must be one of {valid_size_handlers}, "
                f"got {self.handle_input_size}"
            )

    def _initialize_swin_model(
        self, in_channels: int, config: SwinTransformerEncoderConfig
    ) -> None:
        """Initialize the Swin Transformer model from timm."""
        try:
            logger.info(
                f"Attempting to initialize {config.model_name} with "
                f"img_size={self.img_size}"
            )
            self.swin = timm.create_model(
                config.model_name,
                pretrained=config.pretrained,
                in_chans=in_channels,
                features_only=config.features_only,
                out_indices=self.out_indices,
            )
            logger.info(f"Successfully initialized {config.model_name} model")
        except (RuntimeError, ValueError, TypeError, FileNotFoundError) as e:
            logger.error(f"Failed to initialize {config.model_name}: {str(e)}")
            logger.error("Falling back to ResNet-based encoder")
            try:
                fallback_model = "resnet34"
                self.swin = timm.create_model(
                    fallback_model,
                    pretrained=config.pretrained,
                    in_chans=in_channels,
                    features_only=True,
                    out_indices=self.out_indices,
                )
                logger.info(
                    f"Successfully initialized fallback model {fallback_model}"
                )
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

    def _determine_channel_info(self, in_channels: int) -> None:
        """Determine _skip_channels and _out_channels using a dummy pass or
        feature_info."""
        try:
            dummy_input = torch.randn(
                2, in_channels, self.img_size, self.img_size
            )
            self.swin.eval()
            with torch.no_grad():
                dummy_features = self.swin(dummy_input)
            if not isinstance(dummy_features, list) or not dummy_features:
                raise RuntimeError(
                    "Timm model did not return a list of features."
                )
            actual_all_channels = [feat.shape[1] for feat in dummy_features]
            logger.info(
                "Detected feature channels from dummy pass: "
                f"{actual_all_channels}"
            )
            self._skip_channels = list(reversed(actual_all_channels[:-1]))
            self._out_channels = actual_all_channels[-1]
        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            logger.error(
                f"Failed to determine output channels via dummy forward pass: "
                f"{e}. Falling back to timm feature_info."
            )
            feature_info = self.swin.feature_info
            if feature_info:
                all_channels = [info["num_chs"] for info in feature_info.info]
                self._skip_channels = list(reversed(all_channels[:-1]))
                self._out_channels = all_channels[-1]
                logger.warning(
                    f"Using fallback channels from feature_info: skips="
                    f"{self._skip_channels}, out={self._out_channels}"
                )
            else:
                logger.error(
                    "Cannot determine output channels from dummy pass or "
                    "feature_info."
                )
                raise RuntimeError(
                    "Could not determine encoder output channel dimensions."
                ) from e

    def _calculate_reduction_factors(self) -> None:
        """Calculate and store reduction factors for each feature stage."""
        if hasattr(self.swin, "feature_info") and self.swin.feature_info.info:
            self.reduction_factors = [
                info.get("reduction", 2 ** (i + 1))
                for i, info in enumerate(self.swin.feature_info.info)
            ]
        else:
            num_stages = len(self._skip_channels) + 1
            self.reduction_factors = [
                self.patch_size * (2**i) for i in range(num_stages)
            ]
            logger.warning(
                f"Estimating reduction factors: {self.reduction_factors}"
            )

    def __init__(
        self,
        in_channels: int,
        config: SwinTransformerEncoderConfig | None = None,
    ):
        """
        Initialize the Swin Transformer V2 Encoder.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            config (SwinTransformerEncoderConfig, optional): Configuration
                object. If None, default values will be used.
        """
        super().__init__(in_channels)

        if config is None:
            config = SwinTransformerEncoderConfig()

        self.config = config  # Store config for potential later use
        self.in_channels = in_channels  # Store in_channels for helper methods

        self._initialize_attributes(config)
        self._validate_config(config)
        self._initialize_swin_model(in_channels, config)
        self._determine_channel_info(in_channels)
        self._calculate_reduction_factors()

        logger.info(
            "Encoder initialized with out_channels="
            f"{self._out_channels}, skip_channels={self._skip_channels}"
        )

        if self.freeze_layers:
            self._apply_layer_freezing()

    def _validate_model_config(self, model_name: str, img_size: int) -> None:
        """
        Validate that the model name is compatible with the configured image
        size.

        Args:
            model_name (str): The model name to validate.
            img_size (int): The configured image size.
        """
        # Extract the image size from the model name if present
        if "_" in model_name:
            name_parts = model_name.split("_")
            if len(name_parts) >= self.min_model_name_parts_for_size_check:
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

    def _preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """
        Preprocess input tensor based on configuration.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Optional[Dict]]:
                - Preprocessed tensor
                - Dictionary with metadata for potential post-processing, or
                    None
        """
        # Store original size for reference
        original_size = (x.shape[2], x.shape[3])
        metadata: dict[str, Any] = {"original_size": original_size}

        # Basic input validation
        if x.dim() != self.expected_input_dims:
            raise ValueError(
                f"Expected {self.expected_input_dims}D input (B,C,H,W), "
                f"got {x.dim()}D"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )

        if (
            min(x.size(2), x.size(3)) < self.patch_size
            and self.handle_input_size == "none"
        ):
            raise ValueError(
                f"Input dimensions (H={x.size(2)}, W={x.size(3)}) must be at "
                f"least {self.patch_size}x{self.patch_size} with "
                f"handle_input_size='none'"
            )

        # Apply preprocessing based on configuration
        if self.handle_input_size == "resize":
            # Resize to the model's expected size
            if original_size != (self.img_size, self.img_size):
                x = F.interpolate(
                    x,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                metadata["resized"] = True
                logger.debug(
                    f"Resized input from {original_size} to "
                    f"({self.img_size}, {self.img_size})"
                )

        elif self.handle_input_size == "pad":
            # Calculate padding to make dimensions divisible by patch_size
            B, C, H, W = x.shape
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

            if pad_h > 0 or pad_w > 0:
                # Apply padding (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
                metadata["padding"] = (0, pad_w, 0, pad_h)
                metadata["padded_size"] = (H + pad_h, W + pad_w)
                logger.debug(
                    f"Padded input from {(H, W)} to {(H + pad_h, W + pad_w)}"
                )

        return x, metadata

    def _postprocess_features(
        self,
        features: list[torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """
        Apply post-processing to features if needed.

        Args:
            features (List[torch.Tensor]): Feature tensors.
            metadata (Dict[str, Any], optional): Metadata from preprocessing.

        Returns:
            List[torch.Tensor]: Post-processed features.
        """
        # If no metadata or no modifications were made, return as is
        if metadata is None:
            return features

        processed_features = []

        for feature_item in features:  # Renamed loop variable
            processed_feature_item = feature_item  # Start with original
            # Apply post-processing if needed
            if "resized" in metadata and metadata["resized"]:
                # Resize back to original spatial dimensions adjusted for the
                # reduction factor
                original_h, original_w = metadata["original_size"]
                feature_h = max(1, original_h // self.patch_size)
                feature_w = max(1, original_w // self.patch_size)

                # Only resize if dimensions are different
                current_h, current_w = (
                    processed_feature_item.shape[2],
                    processed_feature_item.shape[3],
                )
                if (current_h, current_w) != (feature_h, feature_w):
                    processed_feature_item = F.interpolate(
                        processed_feature_item,  # Use the working variable
                        size=(feature_h, feature_w),
                        mode="bilinear",
                        align_corners=False,
                    )

            # If we need to handle padding, we could add logic here to crop
            # the features back to the unpadded size if needed

            processed_features.append(
                processed_feature_item
            )  # Append the processed version

        return processed_features

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the Swin Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - bottleneck: Final feature map, deepest encoding.
                - skip_connections: List of intermediate feature maps for skip
                                  connections, in order from high to low
                                  resolution (for the decoder).
        """
        # Handle input preprocessing
        x, metadata = self._preprocess_input(x)

        # Forward pass through the model
        features = self.swin(x)

        # Sanity check for dimensions - features should come from low to high
        # resolution
        if len(features) != len(self.out_indices):
            logger.warning(
                f"Expected {len(self.out_indices)} features, but got "
                f"{len(features)}. This may indicate a model configuration "
                f"issue."
            )

        # Apply post-processing if necessary
        if metadata:
            features = self._postprocess_features(features, metadata)

        # Prepare return values - we need to reverse the feature order from
        # Swin's native output
        # The bottleneck is the last feature (lowest resolution)
        bottleneck = features[-1]

        # Skip connections are the first N-1 features (from higher resolution)
        # We reverse to get from high to low resolution (decoder's expected
        # order)
        skip_connections = list(reversed(features[:-1]))

        return bottleneck, skip_connections

    def get_feature_info(self) -> list[dict[str, Any]]:
        """
        Get information about feature maps produced by the encoder.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        feature_info = []
        num_skips = len(self._skip_channels)
        for i in range(num_skips):
            # reduction_factors[0] corresponds to skip_channels[0]
            # (highest res skip)
            idx = i  # Index for reduction factor
            channels = self._skip_channels[i]  # High-res to low-res
            feature_info.append(
                {
                    "channels": channels,
                    "reduction_factor": self.reduction_factors[idx],
                    "stage": idx,  # Use index relative to H->L res skips
                }
            )
        # Add bottleneck info
        bottleneck_stage_idx = num_skips
        feature_info.append(
            {
                "channels": self._out_channels,
                "reduction_factor": self.reduction_factors[
                    bottleneck_stage_idx
                ],
                "stage": bottleneck_stage_idx,
            }
        )
        return feature_info

    @property
    def out_channels(self) -> int:
        """Number of channels in the final output tensor (bottleneck)."""
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        """List of channels for each skip connection (high to low res)."""
        return self._skip_channels

    def _apply_layer_freezing(self) -> None:
        """
        Applies layer freezing based on the freeze_layers configuration.

        This is used for transfer learning to control which parts of the model
        are trainable.
        """
        freeze_patterns = []

        # Convert string input to list of patterns
        if isinstance(self.freeze_layers, bool):
            if self.freeze_layers:
                # Default behavior: freeze all except the last block
                if hasattr(self.swin, "stages"):
                    num_stages = len(self.swin.stages)
                    freeze_patterns = [
                        "patch_embed",
                        *(f"stages.{i}" for i in range(num_stages - 1)),
                    ]
                else:
                    logger.warning(
                        "Could not determine stages in model. \
Using basic freezing."
                    )
                    freeze_patterns = ["patch_embed", "blocks.0", "blocks.1"]
            else:
                # No freezing
                return
        elif isinstance(self.freeze_layers, str):
            if self.freeze_layers.lower() == "all":
                freeze_patterns = [".*"]  # Freeze all layers
            else:
                # Split comma-separated string
                freeze_patterns = [
                    p.strip() for p in self.freeze_layers.split(",")
                ]
        elif isinstance(self.freeze_layers, list):
            freeze_patterns = self.freeze_layers
        else:
            logger.warning(
                f"Unsupported freeze_layers type: {type(self.freeze_layers)}"
            )
            return

        # Apply freezing
        frozen_params = 0
        total_params = 0

        for name, param in self.swin.named_parameters():
            total_params += param.numel()
            # Check if parameter matches any pattern
            if any(re.search(pattern, name) for pattern in freeze_patterns):
                param.requires_grad = False
                frozen_params += param.numel()

        # Log freezing statistics
        frozen_percentage = (frozen_params / max(total_params, 1)) * 100
        logger.info(
            f"Transfer learning configuration applied: "
            f"Froze {frozen_params:,} parameters "
            f"({frozen_percentage:.1f}% of model)"
        )

    def get_optimizer_param_groups(self, base_lr: float = 0.001) -> list[dict]:
        """
        Returns parameter groups with differential learning rates for
        fine-tuning.

        This enables techniques like discriminative learning rates, where
        different parts of the model are trained with different learning
        rates.

        Args:
            base_lr (float): Base learning rate to scale other LRs from

        Returns:
            List[Dict]: Parameter groups with custom learning rates
        """
        if not self.finetune_lr_scale:
            # If no scaling is specified, return a single parameter group
            return [{"params": self.parameters(), "lr": base_lr}]

        # Create parameter groups with scaled learning rates
        param_groups = []
        default_group_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            matched = False
            for pattern, scale in self.finetune_lr_scale.items():
                if re.search(pattern, name):
                    param_groups.append(
                        {
                            "name": pattern,
                            "params": [param],
                            "lr": base_lr * scale,
                        }
                    )
                    matched = True
                    break

            if not matched:
                default_group_params.append(param)

        # Add default group with base learning rate
        if default_group_params:
            param_groups.append(
                {
                    "name": "default",
                    "params": default_group_params,
                    "lr": base_lr,
                }
            )

        # Log parameter group configuration
        logger.info(
            f"Created {len(param_groups)} parameter groups for fine-tuning"
        )
        for group in param_groups:
            if "name" in group:
                logger.debug(
                    f"Parameter group '{group['name']}': LR = {group['lr']}"
                )

        return param_groups

    def _get_patterns_for_epoch(
        self, current_epoch: int, unfreeze_schedule: dict[int, list[str]]
    ) -> list[str]:
        """Determines patterns to unfreeze for the current epoch."""
        patterns_to_unfreeze = []
        for epoch, patterns in sorted(unfreeze_schedule.items()):
            if current_epoch >= epoch:
                patterns_to_unfreeze.extend(patterns)
        return patterns_to_unfreeze

    def _log_block_prefixes_debug(self) -> None:
        """Logs available parameter block prefixes for debugging."""
        param_names = [name for name, _ in self.swin.named_parameters()]
        block_prefixes = set()
        for name in param_names:
            parts = name.split(".")
            if len(parts) > 1:
                block_prefixes.add(parts[0])
        logger.debug(
            f"Available parameter block prefixes: {sorted(block_prefixes)}"
        )

    def _adapt_unfreeze_patterns(
        self, patterns_to_unfreeze: list[str]
    ) -> list[str]:
        """Adapts user-defined unfreeze patterns to internal model naming."""
        adapted_patterns = []
        for pattern in patterns_to_unfreeze:
            if pattern.startswith("stages."):
                stage_num = pattern.split(".")[1]
                adapted_patterns.append(f"layers_{stage_num}")
                adapted_patterns.append(f"stages\\.{stage_num}")
                adapted_patterns.append(f"blocks\\.{stage_num}")
                adapted_patterns.append(f"layers\\.{stage_num}")
            else:
                adapted_patterns.append(pattern)
        logger.info(
            f"Adapting patterns {patterns_to_unfreeze} to {adapted_patterns}"
        )
        return adapted_patterns

    def _unfreeze_parameters_by_patterns(
        self, adapted_patterns: list[str]
    ) -> int:
        """Unfreezes model parameters matching the adapted patterns."""
        unfrozen_count = 0
        for name, param in self.swin.named_parameters():
            if not param.requires_grad:  # Only consider frozen parameters
                should_unfreeze = False
                for adapted_pattern in adapted_patterns:
                    if (
                        re.search(adapted_pattern, name)
                        or adapted_pattern in name
                    ):
                        should_unfreeze = True
                        break
                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen_count += 1
                    logger.debug(f"Unfroze parameter: {name}")
        return unfrozen_count

    def _log_unfreezing_results(
        self,
        current_epoch: int,
        unfrozen_count: int,
        original_patterns: list[str],
        adapted_patterns: list[str],
    ) -> None:
        """Logs the results of the unfreezing operation."""
        if unfrozen_count > 0:
            logger.info(
                f"Epoch {current_epoch}: Unfroze {unfrozen_count} parameters "
                f"matching patterns {original_patterns}"
            )
        else:
            logger.warning(
                "No parameters were unfrozen for patterns "
                f"{original_patterns}. Adapted patterns: {adapted_patterns}"
            )
            logger.warning(
                "This may indicate that the pattern names don't match the "
                "model structure."
            )

    def gradual_unfreeze(
        self, current_epoch: int, unfreeze_schedule: dict[int, list[str]]
    ) -> None:
        """
        Gradually unfreezes layers based on the current epoch and schedule.

        This implements the gradual unfreezing technique for transfer learning,
        where deeper layers are unfrozen later in training.

        Args:
            current_epoch (int): Current training epoch
            unfreeze_schedule (Dict[int, List[str]]): Mapping of epochs to
                patterns to unfreeze
                Example: {5: ['stages.0'], 10: ['stages.1'], 15: ['stages.2']}
                The patterns are automatically adapted to the actual model
                structure
                (e.g., 'stages.0' will also match 'layers_0' in SwinV2 models)
        """
        original_patterns_to_unfreeze = self._get_patterns_for_epoch(
            current_epoch, unfreeze_schedule
        )

        if not original_patterns_to_unfreeze:
            return

        self._log_block_prefixes_debug()

        adapted_patterns = self._adapt_unfreeze_patterns(
            original_patterns_to_unfreeze
        )
        unfrozen_count = self._unfreeze_parameters_by_patterns(
            adapted_patterns
        )

        self._log_unfreezing_results(
            current_epoch,
            unfrozen_count,
            original_patterns_to_unfreeze,
            adapted_patterns,
        )
