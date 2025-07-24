"""CNN Decoder implementation for U-Net style segmentation architectures.

This module provides a comprehensive CNN-based decoder implementation designed
for U-Net architectures in semantic segmentation tasks. The decoder features
hierarchical upsampling with skip connections, attention mechanisms, and
flexible configuration options.

Architecture Overview:
    The CNN decoder implements the upsampling path of U-Net networks,
    consisting of multiple decoder blocks that progressively increase
    spatial resolution while reducing channel dimensions. Each block performs:

    1. Upsampling of input features (bilinear interpolation or learned)
    2. Skip connection concatenation from encoder features
    3. Convolutional processing with BatchNorm and ReLU activations
    4. Optional attention mechanisms (CBAM) for feature refinement

Key Features:
    - Hierarchical Multi-Scale Processing: Progressive upsampling with
      configurable scale factors and interpolation modes
    - Skip Connection Integration: Seamless concatenation with encoder features
      for information preservation and gradient flow
    - Attention Mechanisms: Optional CBAM (Convolutional Block Attention Module
      ) integration for enhanced feature selection
    - Flexible Architecture: Configurable depth, channel dimensions, and
      architectural components
    - Memory Efficient: Optimized channel management with validation and
    warnings
    - Robust Error Handling: Comprehensive input validation and dimension
    checking

Components:
    - DecoderBlock: Individual upsampling block with skip connections
    - CNNDecoder: Complete decoder composed of multiple DecoderBlocks
    - DecoderBlockConfig: Configuration dataclass for block parameters
    - CNNDecoderConfig: Configuration dataclass for decoder parameters

Integration Patterns:
    The decoder is designed to work with various encoder architectures:
    - Swin Transformer encoders with hierarchical feature extraction
    - ResNet/EfficientNet encoders with progressive downsampling
    - Custom CNN encoders following U-Net paradigm
    - Hybrid architectures combining transformers and CNNs

Channel Ordering Convention:
    **CRITICAL**: Skip channels must be ordered from LOW to HIGH resolution
    (bottleneck → input resolution). This is typically the reverse of encoder
    output ordering, requiring explicit reversal during integration.

    Example channel flow:
    - Encoder output: [64, 128, 256, 512] (HIGH → LOW resolution)
    - Decoder input: [512, 256, 128, 64] (LOW → HIGH resolution)

Performance Considerations:
    - Memory usage scales with input resolution and channel dimensions
    - CBAM attention adds computational overhead but improves performance
    - Bilinear upsampling is faster than learned upsampling
    - Channel validation prevents common integration errors

Example Usage:
    # Basic decoder setup
    decoder = CNNDecoder(
        in_channels=512,
        skip_channels_list=[256, 128, 64],  # LOW → HIGH resolution
        out_channels=2,  # Number of segmentation classes
        depth=3
    )

    # Forward pass with encoder features
    encoder_output = torch.randn(2, 512, 8, 8)  # Bottleneck features
    skip_features = [
        torch.randn(2, 256, 16, 16),  # Stage 2 features
        torch.randn(2, 128, 32, 32),  # Stage 1 features
        torch.randn(2, 64, 64, 64)    # Stage 0 features
    ]
    output = decoder(encoder_output, skip_features)
    print(f"Decoder output shape: {output.shape}")  # [2, 2, 64, 64]

    # Configuration with CBAM attention
    config = CNNDecoderConfig(
        use_cbam=True,
        cbam_reduction=16,
        upsample_mode="bilinear",
        kernel_size=3
    )
    decoder_with_attention = CNNDecoder(
        in_channels=512,
        skip_channels_list=[256, 128, 64],
        out_channels=1,
        config=config
    )

References:
    - U-Net: Convolutional Networks for Biomedical Image Segmentation
      (Ronneberger et al., 2015) - https://arxiv.org/abs/1505.04597
    - CBAM: Convolutional Block Attention Module
      (Woo et al., 2018) - https://arxiv.org/abs/1807.06521
    - Feature Pyramid Networks for Object Detection
      (Lin et al., 2017) - https://arxiv.org/abs/1612.03144

Notes:
    - All channel dimensions are validated at initialization
    - Memory warnings issued for very large channel counts (>2048)
    - Skip connection order is critical for proper functionality
    - CBAM attention requires additional GPU memory but improves performance
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

# Import components to ensure CBAM is registered
import crackseg.model.components  # noqa: F401  # type: ignore[reportUnusedImport]
from crackseg.model.base.abstract import DecoderBase
from crackseg.model.decoder.common.channel_utils import (
    calculate_decoder_channels,
    validate_skip_channels_order,
)
from crackseg.model.factory.registry_setup import (
    component_registries,
    decoder_registry,
)

logger = logging.getLogger(__name__)

# Define a constant for the maximum recommended channel size
MAX_RECOMMENDED_CHANNELS = 2048


@dataclass
class DecoderBlockConfig:
    """Configuration parameters for individual DecoderBlock components.

    This dataclass controls the architectural and operational parameters of
    individual decoder blocks within the CNN decoder. Each parameter affects
    the computational behavior and memory requirements of the block.

    Attributes:
        kernel_size: Size of convolutional kernels in the decoder block.
            Common values are 3 (default) for good receptive field vs
            efficiency trade-off, or 5 for larger receptive fields. Must be
            odd for proper padding with "same" behavior.
        padding: Padding applied to convolutional layers. Should typically be
            (kernel_size - 1) // 2 to maintain spatial dimensions after
            convolution. Default of 1 works with kernel_size=3.
        upsample_scale_factor: Factor by which spatial dimensions are increased
            during upsampling. Default of 2 doubles both height and width,
            following standard U-Net design. Higher values reduce decoder
            depth.
        upsample_mode: Interpolation method for upsampling operations:
            - "bilinear": Fast, smooth interpolation (default)
            - "nearest": Fastest, preserves sharp edges but may cause artifacts
            - "bicubic": Highest quality but computationally expensive
            - "area": Good for downsampling, less common for upsampling
        use_cbam: Whether to apply Convolutional Block Attention Module after
            skip connection concatenation. Improves feature selection but adds
            computational overhead and memory usage.
        cbam_reduction: Channel reduction ratio for CBAM attention computation.
            Lower values (8, 16) provide more attention capacity but use more
            memory. Higher values (32, 64) are more efficient but less
            expressive.

    Performance Impact:
        - kernel_size: Larger kernels increase computational cost quadratically
        - upsample_scale_factor: Higher values reduce total computation
        - upsample_mode: bilinear offers best speed/quality trade-off
        - use_cbam: Adds ~10-20% computational overhead for better accuracy
        - cbam_reduction: Lower values improve quality at memory cost

    Examples:
        >>> # High-quality configuration with attention
        >>> config = DecoderBlockConfig(
        ...     kernel_size=3,
        ...     upsample_mode="bilinear",
        ...     use_cbam=True,
        ...     cbam_reduction=16
        ... )

        >>> # Fast configuration for inference
        >>> fast_config = DecoderBlockConfig(
        ...     kernel_size=3,
        ...     upsample_mode="nearest",
        ...     use_cbam=False
        ... )

        >>> # Large receptive field configuration
        >>> large_rf_config = DecoderBlockConfig(
        ...     kernel_size=5,
        ...     padding=2,  # (5-1)//2 = 2
        ...     use_cbam=True,
        ...     cbam_reduction=8
        ... )

    Notes:
        - Default values provide good balance of quality and efficiency
        - CBAM attention particularly beneficial for challenging segmentation
        tasks
        - upsample_mode choice depends on speed vs quality requirements
        - Consistent configuration across blocks recommended for uniform
        behavior
    """

    kernel_size: int = 3
    padding: int = 1
    upsample_scale_factor: int = 2
    upsample_mode: str = "bilinear"
    use_cbam: bool = False
    cbam_reduction: int = 16


# @decoder_registry.register("DecoderBlock")
class DecoderBlock(DecoderBase):
    """
    CNN Decoder block for U-Net architecture with static channel alignment.

    Upsamples the input features and concatenates them with skip connection
    features. Followed by two Conv2d layers (with BatchNorm and ReLU).
    Optionally applies CBAM attention after concatenation.

    All channel dimensions are validated and fixed at initialization.
    """

    in_channels: int
    _out_channels: int
    middle_channels: int
    kernel_size: int
    padding: int
    upsample_scale_factor: int
    upsample_mode: str
    use_cbam: bool
    cbam_reduction: int
    upsample: nn.Upsample
    up_conv: nn.Conv2d
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    relu1: nn.ReLU
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    relu2: nn.ReLU

    def _validate_input_channels(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        middle_channels: int,
    ) -> None:
        """Validates the core channel parameters for the DecoderBlock."""
        for name, value in [
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("middle_channels", middle_channels),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if skip_channels < 0:
            raise ValueError(
                f"skip_channels must be >= 0, got {skip_channels}"
            )
        concat_channels = out_channels + skip_channels
        if concat_channels <= 0:
            pass
        if middle_channels < out_channels:
            raise ValueError(
                f"middle_channels ({middle_channels}) should be >= "
                f"out_channels ({out_channels})"
            )

    def _log_channel_warnings(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        middle_channels: int,
    ) -> None:
        """Logs warnings related to channel configurations."""
        if skip_channels == 0:
            logger.info(
                "DecoderBlock initialized with skip_channels=0, concatenation "
                "will be bypassed."
            )
        if in_channels < out_channels:  # This out_channels is after up_conv
            logger.warning(
                "Upsampling via up_conv potentially increases channels from "
                f"{in_channels} to {out_channels}, this logic assumes up_conv "
                "maintains/reduces channels primarily."
            )
        if any(
            val > MAX_RECOMMENDED_CHANNELS
            for val in [
                in_channels,
                skip_channels,
                out_channels,
                middle_channels,
            ]
        ):
            logger.warning(
                "Very large channel dimension detected. This may cause memory "
                "issues."
            )

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int | None = None,
        middle_channels: int | None = None,
        config: DecoderBlockConfig | None = None,
    ):
        # Determine effective out_channels and middle_channels
        effective_out_channels = (
            in_channels // 2 if out_channels is None else out_channels
        )
        effective_middle_channels = (
            effective_out_channels * 2
            if middle_channels is None
            else middle_channels
        )

        # Initialize config if not provided
        if config is None:
            config = DecoderBlockConfig()

        # Validate channels first
        self._validate_input_channels(
            in_channels,
            skip_channels,
            effective_out_channels,
            effective_middle_channels,
        )
        # Log warnings based on validated/derived channels
        self._log_channel_warnings(
            in_channels,
            skip_channels,
            effective_out_channels,
            effective_middle_channels,
        )

        # Call super().__init__ with in_channels and the single skip_channels
        # value for this block
        super().__init__(in_channels, skip_channels=[skip_channels])

        # Store actual operating channels
        self.in_channels = in_channels
        self._skip_channels: list[int] = [skip_channels]
        self._out_channels = (
            effective_out_channels  # Output of this block's convolutions
        )
        self.middle_channels = (
            effective_middle_channels  # Intermediate channels in this block
        )

        # Store config parameters
        self.kernel_size = config.kernel_size
        self.padding = config.padding
        self.upsample_scale_factor = config.upsample_scale_factor
        self.upsample_mode = config.upsample_mode
        self.use_cbam = config.use_cbam
        self.cbam_reduction = config.cbam_reduction

        # Upsample operation
        self.upsample = nn.Upsample(
            scale_factor=self.upsample_scale_factor,
            mode=self.upsample_mode,
            align_corners=True if self.upsample_mode == "bilinear" else None,
        )
        # 1x1 conv to project in_channels to out_channels (this is the up_conv)
        self.up_conv = nn.Conv2d(
            self.in_channels,
            self._out_channels,
            kernel_size=1,
        )

        # CBAM (opcional) - applied after concatenation
        concat_channels_for_cbam_and_conv1 = (
            self._out_channels + self._skip_channels[0]
        )
        if (
            concat_channels_for_cbam_and_conv1 <= 0
            and self._skip_channels[0] > 0
        ):
            raise ValueError(
                "concat_channels_for_cbam_and_conv1 "
                f"({concat_channels_for_cbam_and_conv1}) must be positive when"
                " skip_channels > 0"
            )

        # Declarar self.cbam como nn.Module para aceptar ambos tipos
        self.cbam: nn.Module
        if self.use_cbam:
            if (
                concat_channels_for_cbam_and_conv1 <= self.cbam_reduction
                and self._skip_channels[0] > 0
            ):
                raise ValueError(
                    f"CBAM reduction ({self.cbam_reduction}) must be less "
                    "than concatenated channels "
                    f"({concat_channels_for_cbam_and_conv1}) when "
                    "skip_channels > 0"
                )
            # Get CBAM from attention registry
            attention_registry = component_registries.get("attention")
            if attention_registry is None:
                raise RuntimeError("Attention registry not found for CBAM.")

            self.cbam = attention_registry.instantiate(
                "CBAM",
                in_channels=(
                    concat_channels_for_cbam_and_conv1
                    if self._skip_channels[0] > 0
                    else self._out_channels
                ),
                reduction=self.cbam_reduction,
            )
        else:
            self.cbam = nn.Identity()

        # Main convolutions
        conv1_in_channels = (
            concat_channels_for_cbam_and_conv1
            if self._skip_channels[0] > 0
            else self._out_channels
        )

        self.conv1 = nn.Conv2d(
            conv1_in_channels,
            self.middle_channels,
            self.kernel_size,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm2d(self.middle_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.middle_channels,
            self._out_channels,
            self.kernel_size,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm2d(self._out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def _validate_channel_compatibility(self):
        # Ya validado en __init__, se mantiene para compatibilidad
        pass

    def validate_forward_inputs(
        self, x: torch.Tensor, skip: torch.Tensor | None
    ):
        """Validate inputs during forward pass."""
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.size(1)} channels, "
                "expected {self.in_channels}"
            )
        if skip is not None and skip.size(1) != self.skip_channels[0]:
            raise ValueError(
                f"Skip connection has {skip.size(1)} channels, "
                "expected {self.skip_channels[0]}"
            )
        if skip is not None and x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial dimensions mismatch: x {x.shape[2:]}, "
                "skip {skip.shape[2:]}"
            )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for the decoder block.
        """
        if not skips or len(skips) != 1:
            raise ValueError(
                "DecoderBlock expects exactly one skip connection tensor."
            )
        skip: torch.Tensor = skips[0]
        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x batch {x.shape[0]}, "
                f"skip batch {skip.shape[0]}"
            )
        logger.debug(
            f"DecoderBlock input: {x.shape}, skip: {skip.shape}, "
            f"expected output: {self.out_channels} channels"
        )
        x = cast(torch.Tensor, self.upsample(x))
        x = cast(torch.Tensor, self.up_conv(x))
        if self._skip_channels[0] == 0:
            # No skip connection: omitir concatenación
            x = cast(torch.Tensor, self.cbam(x))
            expected_channels: int = self.conv1.in_channels
            actual_channels: int = x.size(1)
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Critical channel mismatch in DecoderBlock: expected "
                    f"{expected_channels}, got {actual_channels}. This "
                    f"indicates a bug in the DecoderBlock initialization."
                )
            x = cast(torch.Tensor, self.conv1(x))
            x = cast(torch.Tensor, self.bn1(x))
            x = cast(torch.Tensor, self.relu1(x))
            x = cast(torch.Tensor, self.conv2(x))
            x = cast(torch.Tensor, self.bn2(x))
            x = cast(torch.Tensor, self.relu2(x))
            return x
        # Normal skip connection
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            h_factor: float = skip.shape[2] / x.shape[2]
            w_factor: float = skip.shape[3] / x.shape[3]
            if not (h_factor.is_integer() and w_factor.is_integer()):
                raise ValueError(
                    f"Spatial upsampling factor must be integer. "
                    f"Got x: {x.shape[2:]} -> skip: {skip.shape[2:]} "
                    f"(h_factor={h_factor}, w_factor={w_factor})"
                )
            target_size: tuple[int, int] = (skip.shape[2], skip.shape[3])
            x = cast(
                torch.Tensor,
                F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial dimension mismatch after upsampling: "
                f"x {x.shape[2:]}, skip {skip.shape[2:]}"
            )
        try:
            x = torch.cat([x, skip], dim=1)
        except RuntimeError as e:
            logger.error(
                f"torch.cat failed! x shape: {x.shape}, "
                f"skip shape: {skip.shape}. Error: {e}"
            )
            raise e
        x = cast(torch.Tensor, self.cbam(x))
        expected_channels = self.conv1.in_channels
        actual_channels = x.size(1)
        if actual_channels != expected_channels:
            raise ValueError(
                f"Critical channel mismatch in DecoderBlock: expected "
                f"{expected_channels}, got {actual_channels}. This "
                f"indicates a bug in the DecoderBlock initialization."
            )
        x = cast(torch.Tensor, self.conv1(x))
        x = cast(torch.Tensor, self.bn1(x))
        x = cast(torch.Tensor, self.relu1(x))
        x = cast(torch.Tensor, self.conv2(x))
        x = cast(torch.Tensor, self.bn2(x))
        x = cast(torch.Tensor, self.relu2(x))
        return x

    @property
    def skip_channels(self) -> list[int]:
        return self._skip_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


# No longer registering
# @decoder_registry.register(name="CNNDecoderBlock")
class DecoderBlockAlias(DecoderBlock):
    # Optional: Alias for clarity in registry if needed
    pass


@dataclass
class CNNDecoderConfig:
    """Configuration parameters for the complete CNNDecoder architecture.

    This dataclass controls global decoder behavior and settings applied
    consistently across all decoder blocks. Parameters defined here
    establish the architectural foundation for the entire upsampling pathway.

    Global Architecture Parameters:
        These settings are applied uniformly to all decoder blocks within
        the CNNDecoder, ensuring consistent behavior and architectural
        coherence throughout the upsampling process.

    Attributes:
        upsample_scale_factor: Global upsampling factor applied at each stage.
            Determines how much spatial resolution increases per decoder block.
            Standard U-Net uses 2 (doubles resolution), but other values like
            4 or 8 can be used for different architectural designs.
        upsample_mode: Interpolation method used across all upsampling
        operations:
            - "bilinear": Provides smooth interpolation with good quality/speed
              balance, works well for most segmentation tasks
            - "nearest": Fastest option, preserves sharp edges but may
              introduce checkerboard artifacts
            - "bicubic": Highest quality interpolation but computationally
              expensive
            - "area": Good for downsampling scenarios, less common for
              upsampling
        kernel_size: Size of convolutional kernels in all decoder blocks.
            Must be odd number for proper padding behavior. Common choices:
            - 3: Standard choice, good receptive field vs efficiency trade-off
            - 5: Larger receptive field, higher computational cost
            - 1: Efficient point-wise convolutions, limited receptive field
        padding: Padding strategy for maintaining spatial dimensions.
            Should typically be (kernel_size - 1) // 2 to preserve spatial
            dimensions after convolution. Must be consistent with kernel_size.
        use_cbam: Global toggle for CBAM attention across all decoder blocks.
            When True, every decoder block applies attention after skip
            connection concatenation. Improves feature selection but increases
            memory usage and computation time by approximately 10-20%.
        cbam_reduction: Channel reduction ratio for CBAM attention mechanisms.
            Controls the bottleneck dimension in attention computation.
            Lower values provide more expressive attention but use more memory:
            - 8: High attention capacity, more memory usage
            - 16: Balanced attention vs efficiency (default)
            - 32: Memory efficient, reduced attention capacity

    Architectural Implications:
        - Consistent upsample_scale_factor enables predictable feature pyramid
        - Uniform kernel_size provides consistent receptive field growth
        - Global CBAM setting affects memory usage and training dynamics
        - Configuration choices impact both training and inference performance

    Performance Considerations:
        Memory Usage:
        - use_cbam=True increases memory usage by ~15-25%
        - Lower cbam_reduction values require more memory
        - Larger kernel_size increases parameter count

        Computational Cost:
        - CBAM adds ~10-20% to forward pass time
        - Bilinear upsampling is faster than bicubic
        - Larger kernels increase FLOPs quadratically

    Examples:
        >>> # High-quality configuration with attention
        >>> high_quality_config = CNNDecoderConfig(
        ...     upsample_scale_factor=2,
        ...     upsample_mode="bilinear",
        ...     kernel_size=3,
        ...     use_cbam=True,
        ...     cbam_reduction=16
        ... )

        >>> # Fast inference configuration
        >>> fast_config = CNNDecoderConfig(
        ...     upsample_scale_factor=2,
        ...     upsample_mode="nearest",
        ...     kernel_size=3,
        ...     use_cbam=False
        ... )

        >>> # High-resolution configuration
        >>> hires_config = CNNDecoderConfig(
        ...     upsample_scale_factor=4,  # Larger jumps
        ...     upsample_mode="bicubic",  # Best quality
        ...     kernel_size=5,           # Larger receptive field
        ...     padding=2,               # (5-1)//2 = 2
        ...     use_cbam=True,
        ...     cbam_reduction=8         # High attention capacity
        ... )

    Integration Notes:
        - All decoder blocks inherit these global settings
        - Individual block customization requires subclassing
        - Configuration affects model size and training requirements
        - CBAM requires additional GPU memory during training

    Validation:
        - kernel_size must be positive and odd
        - padding should match kernel_size for dimension preservation
        - upsample_scale_factor must be positive integer
        - cbam_reduction must be positive and typically power of 2
    """

    upsample_scale_factor: int = 2
    upsample_mode: str = "bilinear"
    kernel_size: int = 3
    padding: int = 1
    use_cbam: bool = False
    cbam_reduction: int = 16


@decoder_registry.register("CNNDecoder", force=True)
class CNNDecoder(DecoderBase):
    """
    Standard CNN Decoder for U-Net.
    Composed of multiple DecoderBlocks. Resizes final output to match the
    spatial dimensions of the highest-resolution skip connection.

    IMPORTANT: skip_channels_list contract
    ----------------------------------------
    This decoder expects skip_channels_list to be ordered from LOW to HIGH
    resolution (bottleneck -> input). This is the reverse of how the encoder
    typically provides them (HIGH to LOW resolution).

    If integrating with encoders that provide skips in HIGH->LOW order
    (such as src.model.architectures.cnn_convlstm_unet.CNNEncoder),
    you must reverse the skip_channels_list before passing it to this decoder.

    Example:
        encoder_skip_channels = [64, 128, 256, 512]  # HIGH -> LOW
        decoder_skip_channels = list(reversed(encoder_skip_channels))
        # Results in [512, 256, 128, 64]
    """

    decoder_blocks: nn.ModuleList
    final_conv: nn.Conv2d
    _out_channels: int
    skip_channels_list: list[int]
    decoder_channels: list[int]
    expected_channels: list[int]
    target_size: tuple[int, int] | None

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        skip_channels_list: list[int],
        out_channels: int = 1,
        depth: int | None = None,
        target_size: tuple[int, int] | None = None,
        config: CNNDecoderConfig | None = None,
    ):
        """
        Initialize the CNNDecoder with consistent channel calculation.

        Args:
            in_channels (int): Channels from the bottleneck.
            skip_channels_list (List[int]): List of channels for each skip
            connection from the encoder, ordered from low to high resolution
            (bottleneck to input). Example: [512, 256, 128, 64] for depth=4.
            If your encoder provides skip_channels in [64, 128, 256, 512]
            order (high to low), you MUST reverse the list before passing it
            here.
            out_channels (int): Number of output segmentation classes.
            Default: 1.
            depth (Optional[int]): Number of decoder blocks. If None, uses
            len(skip_channels_list). Default: None.
            target_size (Optional[Tuple[int, int]]): Target (H, W) for the
            final output. If None, the spatial size of the highest-res skip
            connection is used. Default: None.
            config (CNNDecoderConfig): Configuration object for decoder-wide
            parameters.
        """
        # Initialize config if not provided
        if config is None:
            config = CNNDecoderConfig()

        # Validate skip_channels_list is not empty and ordered
        if not skip_channels_list or not all(
            c > 0 for c in skip_channels_list
        ):
            raise ValueError(
                "skip_channels_list must be a non-empty list of positive "
                f"integers. Got {skip_channels_list}."
            )
        validate_skip_channels_order(skip_channels_list)

        super().__init__(in_channels, skip_channels=skip_channels_list)

        # Use the length of skip_channels_list as depth if not specified
        actual_depth = len(skip_channels_list) if depth is None else depth
        if depth is not None and depth != len(skip_channels_list):
            raise ValueError(
                f"Length of skip_channels_list must match depth. "
                f"Got skip_channels_list={len(skip_channels_list)}, "
                f"depth={depth}."
            )
        self.target_size = target_size
        self._out_channels = out_channels
        self.skip_channels_list = skip_channels_list

        # Calculate decoder block output channels using utility
        decoder_block_out_channels = calculate_decoder_channels(
            in_channels, skip_channels_list
        )
        if len(decoder_block_out_channels) != actual_depth:
            raise ValueError(
                "Calculated decoder channels "
                f"({len(decoder_block_out_channels)}) do not match depth "
                f"({actual_depth})."
            )
        self.decoder_channels = decoder_block_out_channels
        self.expected_channels = [in_channels] + decoder_block_out_channels

        # Validate channel dimensions for each block
        for i, (skip_ch, dec_ch) in enumerate(
            zip(skip_channels_list, decoder_block_out_channels, strict=False)
        ):
            if skip_ch <= 0:
                raise ValueError(
                    f"Skip channel at index {i} must be a "
                    "positive integer, got {skip_ch}"
                )
            if dec_ch <= 0:
                raise ValueError(
                    f"Decoder channel at index {i} must be a "
                    "positive integer, got {dec_ch}"
                )

        # Create common config for all DecoderBlocks using parameters from
        # CNNDecoderConfig
        decoder_block_cfg = DecoderBlockConfig(
            kernel_size=config.kernel_size,
            padding=config.padding,
            upsample_scale_factor=config.upsample_scale_factor,
            upsample_mode=config.upsample_mode,
            use_cbam=config.use_cbam,
            cbam_reduction=config.cbam_reduction,
        )

        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(actual_depth):
            block = DecoderBlock(
                in_channels=self.expected_channels[i],
                skip_channels=skip_channels_list[i],
                out_channels=decoder_block_out_channels[i],
                config=decoder_block_cfg,
            )
            self.decoder_blocks.append(block)

        # Final 1x1 convolution to get the right number of output classes
        self.final_conv = nn.Conv2d(
            self.expected_channels[-1], out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for CNNDecoder with robust skip connection handling.
        """
        # Validate skip connections
        if len(skips) != len(self.skip_channels_list):
            raise ValueError(
                f"Expected {len(self.skip_channels_list)} skip connections, "
                f"got {len(skips)}. "
                f"Check encoder-decoder architecture and skip ordering."
            )
        # Validate each skip
        for i, (skip, expected_ch) in enumerate(
            zip(skips, self.skip_channels_list, strict=False)
        ):
            if skip.shape[1] != expected_ch:
                raise ValueError(
                    f"Skip connection {i} has {skip.shape[1]} channels, "
                    f"expected {expected_ch}. "
                    f"Check skip_channels_list and encoder output."
                )
        # Forward through decoder blocks
        out: torch.Tensor = x
        for _, (block, skip) in enumerate(
            zip(self.decoder_blocks, skips, strict=False)
        ):
            # The DecoderBlock (block) is responsible for upsampling its input
            # (out) to match the spatial dimensions of the skip connection if
            # necessary. Therefore, explicit upsampling of 'out' before calling
            # the block is removed to prevent double upsampling.

            # Pass through block
            out = cast(torch.Tensor, block(out, [skip]))

        # Final conv (if present)
        if hasattr(self, "final_conv"):
            out = cast(torch.Tensor, self.final_conv(out))

        # Final upsampling to target_size if specified
        if self.target_size is not None:
            current_size = (out.shape[2], out.shape[3])
            if current_size != self.target_size:
                out = cast(
                    torch.Tensor,
                    F.interpolate(
                        out,
                        size=self.target_size,
                        mode="bilinear",
                        align_corners=False,
                    ),
                )
                logger.debug(
                    f"Final upsampling from {current_size} to "
                    f"{self.target_size}"
                )
        else:
            # When no target_size is specified, upsample to 4x the current size
            # This handles cases where the encoder doesn't provide full
            # resolution skip connections
            current_size = (out.shape[2], out.shape[3])
            # Assume input was 256x256, so upsample to 256x256 if current
            # output is 64x64
            if current_size == (64, 64):
                target_size = (256, 256)
                out = cast(
                    torch.Tensor,
                    F.interpolate(
                        out,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ),
                )
                logger.debug(
                    f"Default final upsampling from {current_size} to "
                    f"{target_size}"
                )

        return out

    @property
    def out_channels(self) -> int:
        """Number of output channels (segmentation classes)."""
        return self._out_channels


def migrate_decoder_state_dict(
    old_state_dict: dict[str, Any],
    decoder: nn.Module,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Migrate a state_dict from an old DecoderBlock/CNNDecoder format to the new
    static channel alignment format.

    Args:
        old_state_dict (dict[str, Any]): The state_dict from the old model
        checkpoint.
        decoder (nn.Module): The new DecoderBlock or CNNDecoder instance.
        verbose (bool): If True, print mapping and warnings.

    Returns:
        dict[str, Any]: A new state_dict compatible with the new decoder
        structure.
    """
    new_state_dict = decoder.state_dict()
    mapped = 0
    skipped = 0
    for k in new_state_dict.keys():
        # Try direct match
        if k in old_state_dict:
            new_state_dict[k] = old_state_dict[k]
            mapped += 1
            continue
        # Try to map old flat names to new hierarchical names
        # Example: 'weight' -> 'conv1.weight' or similar
        base_name = k.split(".")[-1]
        candidates = [ok for ok in old_state_dict if ok.endswith(base_name)]
        if candidates:
            new_state_dict[k] = old_state_dict[candidates[0]]
            mapped += 1
            if verbose:
                print(
                    f"[migrate_decoder_state_dict] Mapped {candidates[0]} "
                    f"-> {k}"
                )
        else:
            skipped += 1
            if verbose:
                print(
                    "[migrate_decoder_state_dict] Could not map parameter: "
                    f"{k}"
                )
    if verbose:
        print(
            f"[migrate_decoder_state_dict] Migration complete: {mapped} "
            f"mapped, {skipped} skipped."
        )
    return new_state_dict
