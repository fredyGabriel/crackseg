import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.model.base.abstract import DecoderBase
from src.model.components.cbam import CBAM
from src.model.decoder.common.channel_utils import (
    calculate_decoder_channels,
    validate_skip_channels_order,
)
from src.model.factory.registry_setup import decoder_registry

logger = logging.getLogger(__name__)

# Define a constant for the maximum recommended channel size
MAX_RECOMMENDED_CHANNELS = 2048


@dataclass
class DecoderBlockConfig:
    """Configuration for the DecoderBlock."""

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
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if not isinstance(skip_channels, int):
            raise TypeError(
                "skip_channels must be an integer, got "
                f"{type(skip_channels).__name__}"
            )
        if skip_channels < 0:
            raise ValueError(
                f"skip_channels must be >= 0, got {skip_channels}"
            )

        concat_channels = (
            out_channels + skip_channels
        )  # Assuming out_channels is derived from upsample(in_channels)
        if concat_channels <= 0:
            # This check might be more complex if up_conv changes out_channels
            # significantly from in_channels
            # For now, assumes out_channels (after up_conv) is a positive value
            # related to in_channels.
            # Silently pass if decoder doesn't have in_channels for now for
            # test mocks
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
        self._skip_channels = [skip_channels]  # type: list[int]
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
        # The output of this up_conv is what gets concatenated with skip
        # connection
        # So, the `out_channels` parameter to this block should reflect its
        # output
        self.up_conv = nn.Conv2d(
            self.in_channels,
            self._out_channels,
            kernel_size=1,  # Output of up_conv is self._out_channels
        )

        # CBAM (opcional) - applied after concatenation
        # Concatenated channels: output of up_conv (self._out_channels) +
        # self._skip_channels
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
                # This check is only relevant if there's a skip connection to
                # concatenate
                raise ValueError(
                    f"CBAM reduction ({self.cbam_reduction}) must be less "
                    "than concatenated channels "
                    f"({concat_channels_for_cbam_and_conv1}) when "
                    "skip_channels > 0"
                )
            self.cbam = CBAM(
                in_channels=(
                    concat_channels_for_cbam_and_conv1
                    if self._skip_channels[0] > 0
                    else self._out_channels
                ),  # CBAM input is different if no skip
                reduction=self.cbam_reduction,
            )
        else:
            self.cbam = nn.Identity()

        # Main convolutions
        # Input to conv1 is different if there's no skip connection
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
        skip = skips[0]
        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x batch {x.shape[0]}, "
                f"skip batch {skip.shape[0]}"
            )
        logger.debug(
            f"DecoderBlock input: {x.shape}, skip: {skip.shape}, "
            f"expected output: {self.out_channels} channels"
        )
        x = self.upsample(x)
        x = self.up_conv(x)
        if self._skip_channels[0] == 0:
            # No skip connection: omitir concatenación
            x = self.cbam(x)
            expected_channels = self.conv1.in_channels
            actual_channels = x.size(1)
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Critical channel mismatch in DecoderBlock: expected "
                    f"{expected_channels}, got {actual_channels}. This "
                    f"indicates a bug in the DecoderBlock initialization."
                )
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            return x
        # Normal skip connection
        target_h, target_w = skip.shape[2:]
        current_h, current_w = x.shape[2:]
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            h_factor = skip.shape[2] / x.shape[2]
            w_factor = skip.shape[3] / x.shape[3]
            if not (h_factor.is_integer() and w_factor.is_integer()):
                raise ValueError(
                    f"Spatial upsampling factor must be integer. "
                    f"Got x: {x.shape[2:]} -> skip: {skip.shape[2:]} "
                    f"(h_factor={h_factor}, w_factor={w_factor})"
                )
            x = F.interpolate(
                x,
                size=(skip.shape[2], skip.shape[3]),
                mode="bilinear",
                align_corners=False,
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
        x = self.cbam(x)
        expected_channels = self.conv1.in_channels
        actual_channels = x.size(1)
        if actual_channels != expected_channels:
            raise ValueError(
                f"Critical channel mismatch in DecoderBlock: expected "
                f"{expected_channels}, got {actual_channels}. This "
                f"indicates a bug in the DecoderBlock initialization."
            )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
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
    """Configuration for the CNNDecoder."""

    upsample_scale_factor: int = 2
    upsample_mode: str = "bilinear"
    kernel_size: int = 3
    padding: int = 1
    use_cbam: bool = False
    cbam_reduction: int = 16


@decoder_registry.register("CNNDecoder")
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
            isinstance(c, int) and c > 0 for c in skip_channels_list
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
            if not isinstance(skip_ch, int) or skip_ch <= 0:
                raise ValueError(
                    f"Skip channel at index {i} must be a "
                    "positive integer, got {skip_ch}"
                )
            if not isinstance(dec_ch, int) or dec_ch <= 0:
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

        Args:
            x (torch.Tensor): Input tensor from bottleneck, shape (B, C, H, W)
            skips (List[torch.Tensor]): List of skip tensors from encoder, "
            "ordered from low to high resolution.
                Each skip must match the expected channel and spatial "
                "dimensions after upsampling.

        Returns:
            torch.Tensor: Output tensor after decoding.

        Raises:
            ValueError: If skip connections are missing, have wrong number, "
            "or mismatched channels/dimensions.

        Contract:
            - Number of skips must match number of decoder blocks.
            - Each skip must have the expected number of channels and spatial
            dimensions after upsampling.
            - Skips must be ordered from low to high resolution (matching
            skip_channels_list).
        """
        # Validate skip connections
        if not isinstance(skips, list | tuple):
            raise TypeError(
                f"skips must be a list or tuple, got {type(skips)}"
            )
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
            if skip is None:
                raise ValueError(
                    f"Skip connection {i} is None. All skips "
                    "must be valid tensors."
                )
            if skip.shape[1] != expected_ch:
                raise ValueError(
                    f"Skip connection {i} has {skip.shape[1]} channels, "
                    "expected {expected_ch}. "
                    f"Check skip_channels_list and encoder output."
                )
        # Forward through decoder blocks
        out = x
        for _, (block, skip) in enumerate(
            zip(self.decoder_blocks, skips, strict=False)
        ):
            # The DecoderBlock (block) is responsible for upsampling its input
            # (out) to match the spatial dimensions of the skip connection if
            # necessary. Therefore, explicit upsampling of 'out' before calling
            # the block is removed to prevent double upsampling.

            # Pass through block
            out = block(out, [skip])
        # Final conv (if present)
        if hasattr(self, "final_conv"):
            out = self.final_conv(out)
        return out

    @property
    def out_channels(self) -> int:
        """Number of output channels (segmentation classes)."""
        return self._out_channels


def migrate_decoder_state_dict(old_state_dict, decoder, verbose=True):
    """
    Migrate a state_dict from an old DecoderBlock/CNNDecoder format to the new
    static channel alignment format.

    Args:
        old_state_dict (dict): The state_dict from the old model checkpoint.
        decoder (nn.Module): The new DecoderBlock or CNNDecoder instance.
        verbose (bool): If True, print mapping and warnings.

    Returns:
        dict: A new state_dict compatible with the new decoder structure.

    Notes:
        - This function attempts to map parameter names from old checkpoints
        to the new static structure.
        - If mapping is not possible, a warning is issued and the parameter is
        skipped.
        - The user should verify the loaded model for correctness.
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
