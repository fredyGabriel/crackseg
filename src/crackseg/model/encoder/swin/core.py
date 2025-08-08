"""Core Swin Transformer V2 Encoder implementation.

This module contains the main SwinTransformerEncoder class that integrates
all the specialized modules for a complete encoder implementation.
"""

import logging
from typing import Any

import torch

from crackseg.model.base import EncoderBase
from crackseg.model.encoder.swin.config import SwinTransformerEncoderConfig
from crackseg.model.encoder.swin.initialization import SwinModelInitializer
from crackseg.model.encoder.swin.preprocessing import SwinPreprocessor
from crackseg.model.encoder.swin.transfer_learning import SwinTransferLearning

logger = logging.getLogger(__name__)


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

    def _initialize_attributes(
        self, config: SwinTransformerEncoderConfig
    ) -> None:
        """Initialize instance attributes from configuration.

        Sets up all encoder instance attributes based on the provided
        configuration object. Attributes are categorized into model
        parameters, training settings, and internal state variables.

        Args:
            config: Configuration object containing all encoder settings.
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

        Args:
            config: Configuration object to validate.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        SwinModelInitializer.validate_model_config(
            config.model_name,
            self.img_size,
            self.min_model_name_parts_for_size_check,
        )
        SwinPreprocessor.validate_config_parameters(self.handle_input_size)

    def _initialize_swin_model(
        self, in_channels: int, config: SwinTransformerEncoderConfig
    ) -> None:
        """Initialize Swin Transformer model with robust error handling.

        Args:
            in_channels: Number of input channels for model creation.
            config: Configuration object containing model specifications.

        Raises:
            ValueError: If both primary and fallback model initialization fail.
        """
        self.swin = SwinModelInitializer.initialize_swin_model(
            in_channels, config, self.out_indices
        )

    def _determine_channel_info(self, in_channels: int) -> None:
        """Determine skip_channels and out_channels using a dummy pass."""
        self._skip_channels, self._out_channels = (
            SwinModelInitializer.determine_channel_info(
                self.swin, in_channels, self.img_size
            )
        )

    def _calculate_reduction_factors(self) -> None:
        """Calculate and store reduction factors for each feature stage."""
        self.reduction_factors = (
            SwinModelInitializer.calculate_reduction_factors(
                self.swin, self._skip_channels, self.patch_size
            )
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the Swin Transformer encoder.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of (bottleneck, skip_connections):
                - bottleneck: Final feature map, deepest encoding.
                - skip_connections: List of intermediate feature maps for skip
                                  connections, in order from high to low
                                  resolution (for the decoder).
        """
        # Handle input preprocessing
        x, metadata = SwinPreprocessor.preprocess_input(
            x,
            self.img_size,
            self.patch_size,
            self.handle_input_size,
            self.expected_input_dims,
            self.in_channels,
        )

        # Forward pass through the model
        features = self.swin(x)

        # Ensure features is a list of tensors
        if not isinstance(features, list):
            features = [features]

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
            features = SwinPreprocessor.postprocess_features(
                features, metadata, self.patch_size
            )

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
        """Get information about feature maps produced by the encoder.

        Returns:
            List of dictionaries containing information about each feature map,
            including channels and reduction factor.
        """
        # Import locally to avoid circular imports
        from crackseg.model.encoder.feature_info_utils import (
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
        """Apply layer freezing based on the freeze_layers configuration."""
        SwinTransferLearning.apply_layer_freezing(
            self.swin, self.freeze_layers
        )

    def get_optimizer_param_groups(
        self, base_lr: float = 0.001
    ) -> list[dict[str, Any]]:
        """Return parameter groups with differential learning rates for
        fine-tuning.

        Args:
            base_lr: Base learning rate to scale other LRs from.

        Returns:
            List of parameter groups with custom learning rates.
        """
        return SwinTransferLearning.get_optimizer_param_groups(
            self.swin, self.finetune_lr_scale, base_lr
        )

    def gradual_unfreeze(
        self, current_epoch: int, unfreeze_schedule: dict[int, list[str]]
    ) -> None:
        """Gradually unfreeze layers based on the current epoch and schedule.

        Args:
            current_epoch: Current training epoch.
            unfreeze_schedule: Mapping of epochs to patterns to unfreeze.
        """
        SwinTransferLearning.gradual_unfreeze(
            self.swin, current_epoch, unfreeze_schedule
        )
