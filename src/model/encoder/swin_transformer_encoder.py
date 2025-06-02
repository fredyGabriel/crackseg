"""Swin Transformer V2 Encoder for semantic segmentation tasks.

This module implements a Swin Transformer V2-based encoder that adapts the
transformer architecture for use in U-Net style segmentation models. It
provides multi-scale feature extraction with hierarchical representations
suitable for dense prediction tasks.

Key Features:
    - Swin Transformer V2 architecture with improved training stability
    - Multi-scale feature extraction for U-Net decoder compatibility
    - Flexible input size handling (resize, pad, or none)
    - Layer freezing and gradual unfreezing for transfer learning
    - Comprehensive error handling with ResNet fallback
    - Optimizer parameter grouping for fine-tuning strategies

Architecture Overview:
    The Swin Transformer uses hierarchical windows that shift between layers
    to enable cross-window connections while maintaining computational
    efficiency. Each stage performs patch merging that reduces spatial
    resolution by 2x while doubling the channel dimensions.

    Typical feature pyramid:
    - Stage 0: H/4 x W/4 x C (patch size 4)
    - Stage 1: H/8 x W/8 x 2C
    - Stage 2: H/16 x W/16 x 4C
    - Stage 3: H/32 x W/32 x 8C

Integration:
    - Designed for U-Net decoder integration with skip connections
    - Compatible with various Swin model variants from timm library
    - Supports both pretrained and randomly initialized models
    - Provides channel information for decoder configuration

Error Handling:
    - Graceful fallback to ResNet encoder if Swin model fails
    - Comprehensive logging for debugging model initialization
    - Robust input size validation and adaptation
    - Channel dimension detection with multiple fallback strategies

Example Usage:
    # Basic encoder initialization
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=True,
        img_size=256
    )
    encoder = SwinTransformerEncoder(in_channels=3, config=config)

    # Forward pass with multi-scale features
    x = torch.randn(2, 3, 256, 256)
    features, skip_features = encoder(x)

    # Access channel information for decoder setup
    out_channels = encoder.out_channels
    skip_channels = encoder.skip_channels

References:
    - Swin Transformer V2: https://arxiv.org/abs/2111.09883
    - timm library: https://github.com/rwightman/pytorch-image-models
    - U-Net architecture: https://arxiv.org/abs/1505.04597
"""

# a highly cohesive, self-contained SwinTransformerEncoder class. All methods
# are tightly coupled to the encoder logic and splitting would reduce clarity.
# This exception is documented as per the coding-preferences guidelines.

import logging
import re
import typing
from dataclasses import dataclass, field
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import EncoderBase

logger = logging.getLogger(__name__)


@dataclass
class SwinTransformerEncoderConfig:
    """Configuration class for SwinTransformerEncoder initialization.

    This dataclass contains all configuration parameters needed to initialize
    and customize a Swin Transformer encoder. It provides sensible defaults
    while allowing fine-grained control over model behavior.

    Configuration Categories:
        - Model Selection: Specify which Swin variant to use
        - Architecture: Control model structure and feature extraction
        - Input Handling: Configure input size adaptation strategies
        - Training: Set up layer freezing and fine-tuning parameters
        - Validation: Parameters for configuration validation

    Attributes:
        model_name: Name of the Swin Transformer model from timm library.
            Common options include:
            - "swinv2_tiny_window16_256": Tiny model with 256x256 input
            - "swinv2_small_window16_256": Small model variant
            - "swinv2_base_window16_256": Base model variant
            - "swinv2_large_window16_256": Large model variant
        pretrained: Whether to load ImageNet pretrained weights. Recommended
            for transfer learning and generally better performance.
        output_hidden_states: Whether to output intermediate hidden states.
            Set to True for multi-scale feature extraction.
        features_only: Whether to extract only feature maps without
            classification head. Should be True for segmentation tasks.
        out_indices: List of stage indices to extract features from.
            Default [0,1,2,3] extracts features from all 4 stages for
            complete feature pyramid.
        img_size: Expected input image size for model initialization.
            Must match the size in model_name for optimal performance.
        patch_size: Size of image patches. Typically 4 for Swin models.
            Determines the initial downsampling factor.
        use_abs_pos_embed: Whether to use absolute position embeddings.
            Recommended for better position awareness.
        output_norm: Whether to apply normalization to output features.
            Helps with training stability and feature consistency.
        handle_input_size: Strategy for handling input size mismatches:
            - "resize": Resize input to match expected size
            - "pad": Pad input with zeros to match expected size
            - "none": Pass input as-is (may cause errors if size mismatches)
        freeze_layers: Specification for layer freezing during training:
            - False: No layers frozen
            - True: All layers frozen
            - str: Pattern to match layer names for freezing
            - list[str]: List of patterns for selective freezing
        finetune_lr_scale: Dictionary mapping layer patterns to learning
            rate scale factors for differential learning rates during
            fine-tuning. None means uniform learning rate.
        min_model_name_parts_for_size_check: Minimum number of parts in
            model name required for automatic size validation.
        expected_input_dims: Expected number of dimensions in input tensor.
            Should be 4 for batched 2D images (N, C, H, W).

    Examples:
        >>> # Basic configuration for tiny model
        >>> config = SwinTransformerEncoderConfig(
        ...     model_name="swinv2_tiny_window16_256",
        ...     pretrained=True,
        ...     img_size=256
        ... )

        >>> # Configuration with layer freezing
        >>> config = SwinTransformerEncoderConfig(
        ...     model_name="swinv2_base_window16_256",
        ...     freeze_layers=["patch_embed", "layers.0"],
        ...     finetune_lr_scale={"layers.3": 1.0, "layers.2": 0.5}
        ... )

        >>> # Configuration for different input handling
        >>> config = SwinTransformerEncoderConfig(
        ...     handle_input_size="pad",
        ...     img_size=512,
        ...     out_indices=[1, 2, 3]  # Skip stage 0
        ... )

    Notes:
        - Model name should include size information for automatic validation
        - Pretrained weights are highly recommended for transfer learning
        - out_indices should be consecutive for proper feature pyramid
        - Layer freezing patterns use regex matching for flexibility
    """

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
    """Swin Transformer V2 Encoder for U-Net style segmentation networks.

    This encoder implements the Swin Transformer V2 architecture adapted for
    semantic segmentation tasks. It provides hierarchical multi-scale feature
    extraction suitable for decoder networks with skip connections, following
    the U-Net paradigm.

    Architecture Details:
        The Swin Transformer uses a hierarchical design with shifted windows
        that enables efficient computation while maintaining cross-window
        connections. Key improvements in V2 include:

        - Post-normalization for improved training stability
        - Scaled cosine attention for better high-resolution performance
        - Log-spaced continuous relative position bias
        - More flexible parameter configurations

    Feature Extraction:
        The encoder extracts features at multiple scales through patch merging
        operations. Each stage reduces spatial resolution by 2x while typically
        doubling channel dimensions:

        - Input: H x W x 3
        - Stage 0: H/4 x W/4 x C (after patch embedding)
        - Stage 1: H/8 x W/8 x 2C (after first patch merging)
        - Stage 2: H/16 x W/16 x 4C
        - Stage 3: H/32 x W/32 x 8C

    Input Handling:
        Supports flexible input size handling strategies:
        - Resize: Automatically resizes input to match model expectations
        - Pad: Pads input with zeros to match expected dimensions
        - None: Passes input unchanged (may cause errors with size mismatch)

    Training Features:
        - Layer freezing for transfer learning scenarios
        - Gradual unfreezing with epoch-based schedules
        - Differential learning rates for fine-tuning
        - Comprehensive parameter grouping for optimizers

    Error Recovery:
        Includes robust fallback mechanisms:
        - Automatic fallback to ResNet if Swin model fails to load
        - Multiple strategies for channel dimension detection
        - Comprehensive error logging for debugging

    Attributes:
        swin: The underlying timm Swin Transformer model instance
        out_indices: Indices of stages to extract features from
        img_size: Expected input image size
        patch_size: Size of image patches for initial embedding
        handle_input_size: Strategy for input size handling
        freeze_layers: Current layer freezing configuration
        finetune_lr_scale: Learning rate scaling factors by layer
        reduction_factors: Downsampling factors for each stage

    Examples:
        >>> # Basic encoder setup
        >>> config = SwinTransformerEncoderConfig(
        ...     model_name="swinv2_tiny_window16_256",
        ...     pretrained=True,
        ...     img_size=256
        ... )
        >>> encoder = SwinTransformerEncoder(in_channels=3, config=config)

        >>> # Forward pass
        >>> x = torch.randn(2, 3, 256, 256)
        >>> features, skip_features = encoder(x)
        >>> print(f"Output features shape: {features.shape}")
        >>> print(f"Skip features shapes: {[s.shape for s in skip_features]}")

        >>> # Access channel information for decoder
        >>> out_channels = encoder.out_channels
        >>> skip_channels = encoder.skip_channels
        >>> print(f"Output channels: {out_channels}")
        >>> print(f"Skip channels: {skip_channels}")

        >>> # Layer freezing for transfer learning
        >>> encoder._apply_layer_freezing()
        >>>
        >>> # Gradual unfreezing during training
        >>> unfreeze_schedule = {
        ...     5: ["layers.3"],
        ...     10: ["layers.2", "layers.3"],
        ...     15: ["layers.1", "layers.2", "layers.3"]
        ... }
        >>> encoder.gradual_unfreeze(current_epoch=10,
        ...                         unfreeze_schedule=unfreeze_schedule)

    Integration:
        Designed to work seamlessly with:
        - U-Net decoders expecting multi-scale features
        - PyTorch Lightning training workflows
        - Hydra configuration management
        - timm model ecosystem

    Performance Considerations:
        - Memory usage scales with input size and model variant
        - Tiny variants suitable for resource-constrained environments
        - Base/Large variants for maximum performance
        - Gradient checkpointing can reduce memory at cost of speed

    Notes:
        - Requires timm library for model instantiation
        - Pretrained weights highly recommended for best performance
        - Input size should match model configuration for optimal results
        - Channel dimensions are automatically detected from model
    """

    swin: nn.Module  # Explicit type annotation for the attribute

    def _initialize_attributes(
        self, config: SwinTransformerEncoderConfig
    ) -> None:
        """Initialize instance attributes from configuration.

        Sets up all encoder instance attributes based on the provided
        configuration object. Attributes are categorized into model
        parameters, training settings, and internal state variables.

        Args:
            config: Configuration object containing all encoder settings.

        Attributes Set:
            Model Configuration:
                - out_indices: Which feature stages to extract
                - img_size: Expected input image dimensions
                - patch_size: Size of image patches for initial embedding
                - handle_input_size: Strategy for input size handling
                - use_abs_pos_embed: Whether to use absolute position
                embeddings
                - output_norm: Whether to normalize output features

            Training Configuration:
                - freeze_layers: Layer freezing specification
                - finetune_lr_scale: Learning rate scaling by layer pattern

            Internal State:
                - _skip_channels: Channel counts for skip connections
                - _out_channels: Output channel count
                - reduction_factors: Downsampling factors per stage

            Validation Parameters:
                - min_model_name_parts_for_size_check: Minimum model name parts
                - expected_input_dims: Expected input tensor dimensions
        """
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
        """Validate configuration parameters for consistency and correctness.

        Performs comprehensive validation of configuration parameters to catch
        potential issues early. Includes model name validation, input size
        checking, and parameter consistency verification.

        Args:
            config: Configuration object to validate.

        Raises:
            ValueError: If any configuration parameter is invalid or
                inconsistent with model requirements.

        Validation Checks:
            - Model name format and size consistency
            - Input size handling strategy validity
            - Parameter ranges and types
            - Configuration consistency across related parameters
        """
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
        """Initialize Swin Transformer model with robust error handling.

        Attempts to create the specified Swin Transformer model from the timm
        library with automatic fallback to ResNet34 if initialization fails.
        Provides comprehensive logging for debugging initialization issues.

        Args:
            in_channels: Number of input channels for model creation.
            config: Configuration object containing model specifications.

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
                f"img_size={self.img_size}"
            )
            self.swin = typing.cast(
                nn.Module,
                timm.create_model(
                    config.model_name,
                    pretrained=config.pretrained,
                    in_chans=in_channels,
                    features_only=config.features_only,
                    out_indices=self.out_indices,
                ),
            )
            logger.info(f"Successfully initialized {config.model_name} model")
        except (RuntimeError, ValueError, TypeError, FileNotFoundError) as e:
            logger.error(f"Failed to initialize {config.model_name}: {str(e)}")
            logger.error("Falling back to ResNet-based encoder")
            try:
                fallback_model = "resnet34"
                self.swin = typing.cast(
                    nn.Module,
                    timm.create_model(
                        fallback_model,
                        pretrained=config.pretrained,
                        in_chans=in_channels,
                        features_only=True,
                        out_indices=self.out_indices,
                    ),
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
                dummy_features = typing.cast(
                    list[torch.Tensor], self.swin(dummy_input)
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
        """Initialize the Swin Transformer V2 Encoder.

        Creates a Swin Transformer encoder adapted for segmentation tasks with
        comprehensive error handling and automatic fallback mechanisms. The
        encoder performs multi-stage initialization including model loading,
        channel detection, and configuration validation.

        Initialization Process:
            1. Initialize parent EncoderBase class
            2. Set up configuration with defaults if needed
            3. Initialize and validate instance attributes
            4. Load Swin Transformer model from timm library
            5. Detect output channel dimensions via dummy forward pass
            6. Calculate reduction factors for each stage
            7. Apply layer freezing if configured

        Args:
            in_channels: Number of input channels. Common values:
                - 3: RGB images
                - 1: Grayscale images
                - 4: RGBA images
            config: Configuration object controlling encoder behavior.
                If None, uses default SwinTransformerEncoderConfig with
                sensible defaults for most use cases.

        Raises:
            ValueError: If model initialization fails for both primary and
                fallback models, or if configuration validation fails.
            RuntimeError: If channel dimension detection fails completely.

        Examples:
            >>> # Basic initialization with RGB input
            >>> encoder = SwinTransformerEncoder(in_channels=3)

            >>> # Custom configuration
            >>> config = SwinTransformerEncoderConfig(
            ...     model_name="swinv2_base_window16_256",
            ...     img_size=512,
            ...     pretrained=True,
            ...     freeze_layers=["patch_embed"]
            ... )
            >>> encoder = SwinTransformerEncoder(in_channels=3, config=config)

            >>> # Check initialization results
            >>> print(f"Output channels: {encoder.out_channels}")
            >>> print(f"Skip channels: {encoder.skip_channels}")

        Notes:
            - Initialization may take several seconds for large models
            - Internet connection required for downloading pretrained weights
            - Fallback to ResNet34 occurs if Swin model fails to load
            - Model is automatically set to training mode after initialization
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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Preprocess input tensor based on configuration.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Dict]:
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
                    antialias=True,
                )
                metadata["resized"] = True
                logger.debug(
                    f"Resized input from {original_size} to "
                    f"({self.img_size}, {self.img_size})"
                )

        elif self.handle_input_size == "pad":
            # Calculate padding to make dimensions divisible by patch_size
            _, _, H, W = x.shape
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

        return typing.cast(tuple[torch.Tensor, dict[str, Any]], (x, metadata))

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

        processed_features: list[torch.Tensor] = []

        for feature_item in features:  # Renamed loop variable
            processed_feature_item: torch.Tensor = (
                feature_item  # Start with original
            )
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
                        processed_feature_item,
                        size=(feature_h, feature_w),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )

            # If we need to handle padding, we could add logic here to crop
            # the features back to the unpadded size if needed

            processed_features.append(processed_feature_item)

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
        features: list[torch.Tensor] = typing.cast(
            list[torch.Tensor], self.swin(x)
        )

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
        bottleneck: torch.Tensor = features[-1]

        # Skip connections are the first N-1 features (from higher resolution)
        # We reverse to get from high to low resolution (decoder's expected
        # order)
        skip_connections: list[torch.Tensor] = list(reversed(features[:-1]))

        return bottleneck, skip_connections

    def get_feature_info(self) -> list[dict[str, Any]]:
        """
        Get information about feature maps produced by the encoder.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        # Import locally to avoid circular imports
        from src.model.encoder.feature_info_utils import (
            create_feature_info_entry,
        )

        feature_info: list[dict[str, Any]] = []
        num_skips = len(self._skip_channels)

        for i in range(num_skips):
            # reduction_factors[0] corresponds to skip_channels[0]
            # (highest res skip)
            idx = i  # Index for reduction factor
            channels = self._skip_channels[i]  # High-res to low-res
            feature_info.append(
                create_feature_info_entry(
                    channels=channels,
                    reduction=self.reduction_factors[idx],
                    stage=idx,
                    name=f"stage_{idx}",
                )
            )

        # Add bottleneck info
        bottleneck_stage_idx = num_skips
        feature_info.append(
            create_feature_info_entry(
                channels=self._out_channels,
                reduction=self.reduction_factors[bottleneck_stage_idx],
                stage=bottleneck_stage_idx,
                name="bottleneck",
            )
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

    @property
    def feature_info(self) -> list[dict[str, Any]]:
        """Information about output features for each stage.

        Returns:
            List of dictionaries, each containing:
                - 'channels': Number of output channels
                - 'reduction': Spatial reduction factor from input
                - 'stage': Stage index
        """
        return self.get_feature_info()

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
        else:
            freeze_patterns = self.freeze_layers

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

    def get_optimizer_param_groups(
        self, base_lr: float = 0.001
    ) -> list[dict[str, Any]]:
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
        param_groups: list[dict[str, Any]] = []
        default_group_params: list[Any] = []

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
        patterns_to_unfreeze: list[str] = []
        for epoch, patterns in sorted(unfreeze_schedule.items()):
            if current_epoch >= epoch:
                patterns_to_unfreeze.extend(patterns)
        return patterns_to_unfreeze

    def _log_block_prefixes_debug(self) -> None:
        """Logs available parameter block prefixes for debugging."""
        param_names: list[str] = [
            name for name, _ in self.swin.named_parameters()
        ]
        block_prefixes: set[str] = set()
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
        adapted_patterns: list[str] = []
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
