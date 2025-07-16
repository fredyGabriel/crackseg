"""
Hybrid U-Net architecture combining SwinV2 encoder, ASPP bottleneck, and CNN.

This class implements a state-of-the-art hybrid segmentation architecture
that combines the hierarchical feature extraction capabilities of Swin
Transformer V2 with the multi-scale context modeling of ASPP and the
precise localization of CNN decoders.

Architecture Overview:
    The SwinV2CnnAsppUNet represents a state-of-the-art approach to semantic
    segmentation by integrating three specialized components:

    1. SwinV2 Encoder: Hierarchical vision transformer for multi-scale feature
       extraction with shifted window attention mechanisms
    2. ASPP Bottleneck: Atrous Spatial Pyramid Pooling for capturing
       multi-scale contextual information at the bottleneck
    3. CNN Decoder: Traditional convolutional decoder with skip connections and
       optional CBAM attention for precise localization

Key Features:
    - Hybrid Architecture: Combines transformer and CNN strengths for optimal
      performance on segmentation tasks
    - Multi-Scale Processing: Hierarchical feature extraction with ASPP for
      comprehensive context understanding
    - Flexible Configuration: Highly configurable components with independent
      parameter control
    - Skip Connections: U-Net style skip connections for preserving
      fine-grained spatial information
    - Attention Mechanisms: Optional CBAM attention in decoder blocks for
      enhanced feature selection
    - Adaptive Output: Configurable final activation functions for different
      segmentation scenarios

Component Integration:
    The architecture follows a carefully designed data flow:

    Input → SwinV2 Encoder → ASPP Bottleneck → CNN Decoder → Output
              ↓ (skip connections)                ↑
              └─────────────────────────────────────┘

    Channel Flow Example:
    - Input: [B, 3, 256, 256]
    - Encoder stages: [B, 96, 64, 64] → [B, 192, 32, 32] → [B, 384, 16, 16]
      → [B, 768, 8, 8]
    - ASPP bottleneck: [B, 768, 8, 8] → [B, 256, 8, 8]
    - Decoder stages: [B, 256, 8, 8] → [B, 128, 16, 16] → [B, 64, 32, 32]
      → [B, 32, 64, 64]
    - Final output: [B, num_classes, 256, 256]

Performance Characteristics:
    - Memory Efficient: Optimized channel management and attention mechanisms
    - Scalable: Supports various input resolutions and model sizes
    - Robust: Comprehensive error handling and validation
    - Fast Inference: Efficient implementation with minimal overhead

Use Cases:
    - Medical Image Segmentation: Precise boundary detection in medical imaging
    - Satellite Image Analysis: Land cover classification and change detection
    - Autonomous Driving: Road scene understanding and object segmentation
    - Industrial Inspection: Defect detection and quality control

Example Usage:
    # Basic configuration for crack segmentation
    encoder_cfg = {
        "model_name": "swinv2_tiny_window16_256",
        "pretrained": True,
        "img_size": 256,
        "in_channels": 3
    }

    bottleneck_cfg = {
        "out_channels": 256,
        "atrous_rates": [1, 6, 12, 18],
        "dropout_rate": 0.1
    }

    decoder_cfg = {
        "use_cbam": True,
        "cbam_reduction": 16,
        "upsample_mode": "bilinear"
    }

    # Initialize model
    model = SwinV2CnnAsppUNet(
        encoder_cfg=encoder_cfg,
        bottleneck_cfg=bottleneck_cfg,
        decoder_cfg=decoder_cfg,
        num_classes=2,  # Background + crack
        final_activation="softmax"
    )

    # Forward pass
    x = torch.randn(4, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")  # [4, 2, 256, 256]

Configuration Guidelines:
    Encoder Configuration:
    - model_name: Choose based on computational budget and accuracy
      requirements
    - img_size: Must match input image dimensions
    - pretrained: Recommended for better convergence

    Bottleneck Configuration:
    - out_channels: Balance between capacity and efficiency
    - atrous_rates: Adjust based on object scale in target domain
    - dropout_rate: Regularization strength

    Decoder Configuration:
    - use_cbam: Enable for challenging segmentation tasks
    - upsample_mode: "bilinear" for speed, "bicubic" for quality

Integration Notes:
    - Automatic channel dimension inference between components
    - Skip connection ordering handled automatically
    - Target output size derived from encoder configuration
    - Component compatibility validated during initialization

References:
    - Swin Transformer V2: https://arxiv.org/abs/2111.09883
    - DeepLab v3+: https://arxiv.org/abs/1802.02611
    - U-Net: https://arxiv.org/abs/1505.04597
    - CBAM: https://arxiv.org/abs/1807.06521

Notes:
    - Requires sufficient GPU memory for transformer operations
    - Input size should be divisible by patch size for optimal performance
    - Skip connection order is automatically handled during initialization
    - Final activation choice depends on loss function and task requirements
"""

# src/model/architectures/swinv2_cnn_aspp_unet.py

import logging
from typing import Any, cast

import torch
from torch import nn

# Base class
from crackseg.model.base.abstract import BottleneckBase, DecoderBase, UNetBase
from crackseg.model.components.aspp import ASPPModule
from crackseg.model.decoder.cnn_decoder import CNNDecoder

# Components
from crackseg.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter

# Activation factory if needed
# from crackseg.model.factory import create_activation # Assuming exists

logger = logging.getLogger(__name__)


class SwinV2CnnAsppUNet(UNetBase):
    """
    Hybrid U-Net architecture combining SwinV2 encoder, ASPP bottleneck, and
    CNN decoder.

    This class implements a state-of-the-art hybrid segmentation architecture
    that combines the hierarchical feature extraction capabilities of Swin
    Transformer V2 with the multi-scale context modeling of ASPP and the
    precise localization of CNN decoders.

    Architecture Components:
        1. SwinV2 Encoder: Hierarchical vision transformer that extracts
           multi-scale features using shifted window attention. Provides
           rich semantic representations with global context understanding.

        2. ASPP Bottleneck: Atrous Spatial Pyramid Pooling module that captures
           multi-scale contextual information through parallel atrous
           convolutions with different dilation rates.

        3. CNN Decoder: Traditional convolutional decoder with skip connections
           that progressively upsamples features while incorporating encoder
           features for precise boundary localization.

    Key Advantages:
        - Global Context: Transformer encoder captures long-range dependencies
        - Multi-Scale Features: ASPP provides comprehensive scale coverage
        - Precise Localization: CNN decoder with skip connections preserves
          fine-grained spatial information
        - Flexible Configuration: Independent control over each component
        - Memory Efficient: Optimized implementation with automatic channel
          dimension management

    Data Flow:
        Input Image → SwinV2 Encoder → ASPP Bottleneck → CNN Decoder
        → Segmentation Map
                           ↓ (skip connections)              ↑
                           └─────────────────────────────────┘

    Channel Management:
        The architecture automatically handles channel dimension compatibility
        between components:
        - Encoder output channels → Bottleneck input channels
        - Bottleneck output channels → Decoder input channels
        - Encoder skip channels → Decoder skip channels (reversed order)

    Skip Connection Handling:
        Skip connections are automatically reordered from encoder output
        (HIGH→LOW resolution) to decoder input (LOW→HIGH resolution) to
        maintain proper U-Net connectivity.

    Attributes:
        encoder: SwinV2EncoderAdapter instance for feature extraction
        bottleneck: ASPPModule instance for multi-scale context modeling
        decoder: CNNDecoder instance for upsampling and localization
        final_activation_layer: Optional activation function for output

    Examples:
        >>> # Basic binary segmentation setup
        >>> encoder_cfg = {
        ...     "model_name": "swinv2_tiny_window16_256",
        ...     "pretrained": True,
        ...     "img_size": 256,
        ...     "in_channels": 3
        ... }
        >>> bottleneck_cfg = {
        ...     "out_channels": 256,
        ...     "atrous_rates": [1, 6, 12, 18]
        ... }
        >>> decoder_cfg = {
        ...     "use_cbam": True,
        ...     "upsample_mode": "bilinear"
        ... }
        >>> model = SwinV2CnnAsppUNet(
        ...     encoder_cfg=encoder_cfg,
        ...     bottleneck_cfg=bottleneck_cfg,
        ...     decoder_cfg=decoder_cfg,
        ...     num_classes=1,
        ...     final_activation="sigmoid"
        ... )

        >>> # Multi-class segmentation
        >>> model_multiclass = SwinV2CnnAsppUNet(
        ...     encoder_cfg=encoder_cfg,
        ...     bottleneck_cfg=bottleneck_cfg,
        ...     decoder_cfg=decoder_cfg,
        ...     num_classes=5,
        ...     final_activation="softmax"
        ... )

        >>> # Forward pass
        >>> x = torch.randn(2, 3, 256, 256)
        >>> output = model(x)
        >>> print(f"Output shape: {output.shape}")  # [2, 1, 256, 256]

    Performance Considerations:
        - Memory usage scales with input resolution and model size
        - Transformer encoder requires more memory than CNN alternatives
        - ASPP bottleneck adds computational overhead but improves accuracy
        - Skip connections preserve memory throughout forward pass

    Integration Notes:
        - Inherits from UNetBase for standardized interface
        - Component compatibility automatically validated
        - Target output size derived from encoder configuration
        - Supports various activation functions for different tasks

    Notes:
        - Requires sufficient GPU memory for transformer operations
        - Input dimensions should be compatible with patch size
        - Pretrained weights recommended for better convergence
        - Final activation choice should match loss function requirements
    """

    def __init__(
        self,
        encoder_cfg: dict[str, Any],
        bottleneck_cfg: dict[str, Any],
        decoder_cfg: dict[str, Any],
        num_classes: int = 1,
        # e.g., 'sigmoid', 'softmax', None
        final_activation: str | None = "sigmoid",
    ):
        """Initialize the SwinV2CnnAsppUNet hybrid architecture.

        Creates and configures all architecture components with automatic
        channel dimension inference and compatibility validation. The
        initialization process handles component integration, skip connection
        ordering, and output size configuration.

        Initialization Process:
            1. Configure and instantiate SwinV2 encoder with specified
               parameters
            2. Create ASPP bottleneck with encoder-compatible input channels
            3. Set up CNN decoder with bottleneck-compatible input and reversed
               skip connections
            4. Initialize UNetBase with validated component compatibility
            5. Configure final activation layer based on task requirements

        Args:
            encoder_cfg: Configuration dictionary for SwinV2EncoderAdapter.
                Required keys depend on encoder implementation. Common keys:
                - "model_name": Swin model variant (e.g.,
                  "swinv2_tiny_window16_256")
                - "pretrained": Whether to use ImageNet pretrained weights
                - "img_size": Input image size for position embeddings
                - "in_channels": Number of input channels (defaults to 3)
            bottleneck_cfg: Configuration dictionary for ASPPModule.
                Common keys:
                - "out_channels": Output channel dimension
                - "atrous_rates": List of dilation rates for parallel branches
                - "dropout_rate": Dropout probability for regularization
            decoder_cfg: Configuration dictionary for CNNDecoder.
                Common keys:
                - "use_cbam": Whether to use CBAM attention in decoder blocks
                - "cbam_reduction": Channel reduction ratio for CBAM
                - "upsample_mode": Interpolation mode ("bilinear", "nearest",
                  etc.)
                - "kernel_size": Convolution kernel size
            num_classes: Number of output segmentation classes.
                - 1: Binary segmentation (background/foreground)
                - >1: Multi-class segmentation
            final_activation: Final activation function name or None.
                - "sigmoid": For binary segmentation with BCE loss
                - "softmax": For multi-class segmentation with CE loss
                - None: Raw logits output for custom loss functions

        Raises:
            ValueError: If configuration parameters are invalid or
                incompatible.
            KeyError: If required configuration keys are missing.

        Examples:
            >>> # Binary crack segmentation
            >>> encoder_cfg = {
            ...     "model_name": "swinv2_tiny_window16_256",
            ...     "pretrained": True,
            ...     "img_size": 256
            ... }
            >>> bottleneck_cfg = {"out_channels": 256}
            >>> decoder_cfg = {"use_cbam": True}
            >>> model = SwinV2CnnAsppUNet(
            ...     encoder_cfg=encoder_cfg,
            ...     bottleneck_cfg=bottleneck_cfg,
            ...     decoder_cfg=decoder_cfg,
            ...     num_classes=1,
            ...     final_activation="sigmoid"
            ... )

            >>> # Multi-class segmentation with custom configuration
            >>> encoder_cfg = {
            ...     "model_name": "swinv2_base_window16_256",
            ...     "pretrained": True,
            ...     "img_size": 512,
            ...     "in_channels": 3
            ... }
            >>> bottleneck_cfg = {
            ...     "out_channels": 512,
            ...     "atrous_rates": [1, 6, 12, 18],
            ...     "dropout_rate": 0.1
            ... }
            >>> decoder_cfg = {
            ...     "use_cbam": True,
            ...     "cbam_reduction": 16,
            ...     "upsample_mode": "bilinear"
            ... }
            >>> model = SwinV2CnnAsppUNet(
            ...     encoder_cfg=encoder_cfg,
            ...     bottleneck_cfg=bottleneck_cfg,
            ...     decoder_cfg=decoder_cfg,
            ...     num_classes=5,
            ...     final_activation="softmax"
            ... )

        Notes:
            - Channel dimensions are automatically inferred between components
            - Skip connection ordering is handled automatically
            - Target output size derived from encoder img_size parameter
            - Component compatibility validated during UNetBase initialization
        """
        # 1. Instantiate Encoder
        if "in_channels" not in encoder_cfg:
            encoder_cfg["in_channels"] = 3
            logger.warning(
                "encoder_cfg missing 'in_channels', defaulting to 3."
            )
        # *** Get target image size from encoder config ***
        # Default if not specified
        target_img_size = encoder_cfg.get("img_size", 256)
        logger.info(
            "Target output spatial size set to: "
            f"{target_img_size}x{target_img_size}"
        )
        encoder = SwinV2EncoderAdapter(**encoder_cfg)

        # 2. Instantiate Bottleneck
        # Infer in_channels from encoder output
        bottleneck_in_channels = encoder.out_channels
        # ASPPModule inherits from BottleneckBase, cast for type checker
        bottleneck: BottleneckBase = cast(
            BottleneckBase,
            ASPPModule(in_channels=bottleneck_in_channels, **bottleneck_cfg),
        )

        # 3. Instantiate Decoder
        # Infer in_channels from bottleneck output
        # Infer skip_channels_list from encoder output (high-res to low-res)
        decoder_in_channels = bottleneck.out_channels
        # IMPORTANT: Reverse skip channels for the decoder
        # Skip channels must go from LOW->HIGH resolution
        decoder_skip_channels = list(reversed(encoder.skip_channels))

        logger.info(
            f"Encoder skip channels (HIGH->LOW): {encoder.skip_channels}"
        )
        logger.info(
            f"Decoder skip channels (LOW->HIGH): {decoder_skip_channels}"
        )

        # *** FIX: Derive decoder depth from the number of skip connections ***
        decoder_depth = len(decoder_skip_channels)

        # Remove 'depth' from decoder_cfg if present, as it's now derived
        if "depth" in decoder_cfg:
            logger.debug(
                "Ignoring 'depth' in decoder_cfg, deriving from encoder skips."
            )
            # Create a copy to avoid modifying the original config object if
            # passed by ref
            decoder_cfg_copy = {
                k: v for k, v in decoder_cfg.items() if k != "depth"
            }
        else:
            decoder_cfg_copy = decoder_cfg

        # CNNDecoder inherits from DecoderBase, cast for type checker
        decoder: DecoderBase = cast(
            DecoderBase,
            CNNDecoder(
                in_channels=decoder_in_channels,
                skip_channels_list=decoder_skip_channels,
                out_channels=num_classes,
                # *** Pass target size ***
                target_size=(target_img_size, target_img_size),
                depth=decoder_depth,  # *** Pass the derived depth ***
                **decoder_cfg_copy,  # Pass other config params
            ),
        )

        # 4. Initialize UNetBase with the components
        # This call also validates component compatibility
        super().__init__(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )

        # 5. Optional Final Activation
        self.final_activation_layer: nn.Module
        if final_activation:
            if final_activation.lower() == "sigmoid":
                self.final_activation_layer = nn.Sigmoid()
            elif final_activation.lower() == "softmax":
                self.final_activation_layer = nn.Softmax(dim=1)
            # Add other activations if needed (e.g., using create_activation)
            # elif create_activation is not None:
            #     self.final_activation_layer = create_activation(
            # final_activation)
            else:
                raise ValueError(
                    f"Unsupported final_activation: {final_activation}"
                )
        else:
            self.final_activation_layer = nn.Identity()

        logger.info("SwinV2CnnAsppUNet initialized successfully.")
        logger.info(f" Encoder: {encoder}")
        logger.info(f" Bottleneck: {bottleneck}")
        logger.info(f" Decoder: {decoder}")
        logger.info(f" Final Activation: {final_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through the complete hybrid U-Net architecture.

        Processes input images through the three-stage pipeline: encoder for
        feature extraction, bottleneck for multi-scale context modeling, and
        decoder for upsampling and localization. Handles skip connection
        reordering and applies final activation automatically.

        Forward Pass Pipeline:
            1. Encoder Stage: Extract hierarchical features and skip
               connections using SwinV2 transformer with shifted window
               attention
            2. Bottleneck Stage: Apply ASPP for multi-scale context modeling
               with parallel atrous convolutions
            3. Decoder Stage: Progressive upsampling with skip connection
               integration for precise boundary localization
            4. Final Activation: Apply task-specific activation function

        Data Flow:
            Input → SwinV2 Encoder → ASPP Bottleneck → CNN Decoder → Output
                      ↓ (skip connections)              ↑
                      └─────────────────────────────────┘

        Args:
            x: Input tensor representing a batch of images.
                Shape: (batch_size, in_channels, height, width)
                - batch_size: Number of images in the batch
                - in_channels: Number of input channels (typically 3 for RGB)
                - height, width: Spatial dimensions of input images

        Returns:
            Output tensor containing segmentation predictions.
            Shape: (batch_size, num_classes, height, width)
            - batch_size: Same as input batch size
            - num_classes: Number of segmentation classes
            - height, width: Same as input spatial dimensions

        Examples:
            >>> # Binary segmentation forward pass
            >>> model = SwinV2CnnAsppUNet(
            ...     encoder_cfg={"model_name": "swinv2_tiny_window16_256"},
            ...     bottleneck_cfg={"out_channels": 256},
            ...     decoder_cfg={},
            ...     num_classes=1,
            ...     final_activation="sigmoid"
            ... )
            >>> x = torch.randn(4, 3, 256, 256)  # Batch of 4 RGB images
            >>> output = model(x)
            >>> print(f"Output shape: {output.shape}")  # [4, 1, 256, 256]
            >>> print(f"Output range: [{output.min():.3f}, "
            ...       f"{output.max():.3f}]")

            >>> # Multi-class segmentation
            >>> model_multiclass = SwinV2CnnAsppUNet(
            ...     encoder_cfg={"model_name": "swinv2_tiny_window16_256"},
            ...     bottleneck_cfg={"out_channels": 256},
            ...     decoder_cfg={},
            ...     num_classes=5,
            ...     final_activation="softmax"
            ... )
            >>> output_multiclass = model_multiclass(x)
            >>> print(f"Output shape: {output_multiclass.shape}")
            ... # [4, 5, 256, 256]
            >>> # Verify softmax probabilities sum to 1
            >>> print(f"Probability sum: "
            ...       f"{output_multiclass.sum(dim=1).mean():.3f}")

        Raises:
            RuntimeError: If component initialization failed or tensor shapes
                are incompatible during forward pass.
            AssertionError: If encoder, bottleneck, or decoder components are
                not properly initialized.

        Performance Notes:
            - Memory usage peaks during encoder forward pass due to attention
              mechanisms and skip connection storage
            - Computational complexity scales with input resolution and model
              size
            - Skip connection reordering adds minimal overhead
            - Final activation is applied in-place when possible

        Implementation Details:
            - Skip connections are automatically reordered from encoder output
              (HIGH→LOW resolution) to decoder input (LOW→HIGH resolution)
            - Component assertions ensure proper initialization
            - Error handling provides informative messages for debugging
            - Output tensor maintains input spatial dimensions
        """
        assert self.encoder is not None, "Encoder is not initialized."
        assert self.bottleneck is not None, "Bottleneck is not initialized."
        assert self.decoder is not None, "Decoder is not initialized."

        # Pass input through encoder to get bottleneck features and skips
        bottleneck_features, skip_connections = self.encoder(x)

        # Pass bottleneck features directly through the bottleneck module
        bottleneck_output = self.bottleneck(bottleneck_features)

        # Pass bottleneck output and skip connections through the decoder
        # IMPORTANT: The decoder expects skip_connections in LOW->HIGH res
        # order, but the encoder provides HIGH->LOW. We reverse the order.
        reversed_skips = list(reversed(skip_connections))

        decoder_output = self.decoder(bottleneck_output, reversed_skips)

        # Apply final activation
        output = self.final_activation_layer(decoder_output)
        return output


# TODO: Add registration logic if needed by a factory
# from crackseg.model.model_registry import register_model
# register_model("swinv2_cnn_aspp_unet", SwinV2CnnAsppUNet)
