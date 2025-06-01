# src/model/encoder/swin_v2_adapter.py
import logging
from dataclasses import dataclass
from typing import Any

import torch

# Corrected imports based on file search and class definitions
from src.model.base import EncoderBase
from src.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
    SwinTransformerEncoderConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SwinV2AdapterConfig:
    """Configuration for SwinV2EncoderAdapter."""

    model_name: str = "swinv2_tiny_window16_256"
    pretrained: bool = True
    out_indices: list[int] | None = None
    img_size: int = 256
    patch_size: int = 4
    handle_input_size: str = "resize"
    freeze_layers: bool | str | list[str] = False
    finetune_lr_scale: dict[str, float] | None = None
    min_model_name_parts_for_size_check: int = 3
    expected_input_dims: int = 4
    expected_bottleneck_ndim_4d: int = 4  # New field for 4D bottleneck ndim
    expected_bottleneck_ndim_3d: int = 3  # New field for 3D bottleneck ndim


class SwinV2EncoderAdapter(EncoderBase):
    """
    Adapter class for the Swin Transformer V2 encoder (from Task 17)
    to integrate with the U-Net architecture based on EncoderBase.

    This adapter wraps the SwinTransformerEncoder implementation and ensures
    it conforms to the EncoderBase interface.
    """

    def __init__(
        self,
        in_channels: int,
        adapter_config: SwinV2AdapterConfig | None = None,
        # Kwargs for potential EncoderBase needs or future flexibility
        **kwargs: Any,
    ):
        """
        Initializes the SwinV2EncoderAdapter.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            adapter_config (SwinV2AdapterConfig, optional): Configuration for
                the adapter and underlying SwinTransformerEncoder.
                If None, default values will be used.
            **kwargs: Additional keyword arguments (currently unused).
        """
        # Pass in_channels to EncoderBase constructor
        super().__init__(in_channels)

        if adapter_config is None:
            adapter_config = SwinV2AdapterConfig()

        self.adapter_config = adapter_config  # Store adapter_config
        self.expected_bottleneck_ndim_4d = (
            adapter_config.expected_bottleneck_ndim_4d
        )
        self.expected_bottleneck_ndim_3d = (
            adapter_config.expected_bottleneck_ndim_3d
        )

        # Create the SwinTransformerEncoderConfig from adapter_config
        # This ensures that if SwinTransformerEncoder's defaults change,
        # we are robust, but primarily we pass through adapter settings.
        # If specific SwinTransformerEncoder params are not in AdapterConfig,
        # they will take SwinTransformerEncoder's defaults.
        swin_specific_config = SwinTransformerEncoderConfig(
            model_name=adapter_config.model_name,
            pretrained=adapter_config.pretrained,
            output_hidden_states=True,  # Required to get skips
            features_only=True,  # Required for feature maps
            out_indices=(
                adapter_config.out_indices
                if adapter_config.out_indices is not None
                else [0, 1, 2, 3]
            ),  # Default Swin out_indices
            img_size=adapter_config.img_size,
            patch_size=adapter_config.patch_size,
            use_abs_pos_embed=True,  # Default for SwinTEnc
            output_norm=True,  # Default for SwinTEnc
            handle_input_size=adapter_config.handle_input_size,
            freeze_layers=adapter_config.freeze_layers,
            finetune_lr_scale=adapter_config.finetune_lr_scale,
            min_model_name_parts_for_size_check=adapter_config.min_model_name_parts_for_size_check,
            expected_input_dims=adapter_config.expected_input_dims,
        )

        # Instantiate the actual SwinTransformerEncoder from Task 17
        self.encoder = SwinTransformerEncoder(
            in_channels=in_channels,
            config=swin_specific_config,  # Pass the constructed config object
        )

        # Properties (_out_channels, _skip_channels) are handled by the
        # underlying SwinTransformerEncoder and exposed via @property methods
        # below

    @property
    def out_channels(self) -> int:
        """Output channels for the final feature map (bottleneck)."""
        # Directly use the property from the wrapped encoder
        return self.encoder.out_channels

    @property
    def skip_channels(self) -> list[int]:
        """List of channels for skip connections (high to low resolution)."""
        # Directly use the property from the wrapped encoder
        return self.encoder.skip_channels

    def _reshape_4d_bottleneck(
        self, bottleneck_features: torch.Tensor
    ) -> torch.Tensor:
        """Handles reshaping for 4D bottleneck features."""
        expected_channels = self.encoder.out_channels
        if bottleneck_features.shape[1] == expected_channels:
            # Shape is likely (B, C, H, W) - Correct format
            pass
        elif bottleneck_features.shape[3] == expected_channels:
            # Shape is likely (B, H, W, C) - Permute to (B, C, H, W)
            bottleneck_features = bottleneck_features.permute(
                0, 3, 1, 2
            ).contiguous()
            logger.debug(
                f"Permuted bottleneck features to {bottleneck_features.shape}"
            )
        else:
            logger.warning(
                f"Bottleneck features have unexpected shape: "
                f"{bottleneck_features.shape}. Expected channels "
                f"{expected_channels} at dim 1 or 3."
            )
        return bottleneck_features

    def _determine_hw_for_3d_reshape(
        self, L: int, original_input_x: torch.Tensor
    ) -> tuple[int, int]:
        """Determines Height and Width for reshaping a 3D bottleneck tensor."""
        h, w = -1, -1
        if hasattr(self.encoder.swin, "patch_embed") and hasattr(
            self.encoder.swin.patch_embed, "grid_size"
        ):
            grid_size = self.encoder.swin.patch_embed.grid_size
            h, w = grid_size[0], grid_size[1]
        elif (
            hasattr(self.encoder, "reduction_factors")
            and self.encoder.reduction_factors
        ):
            final_reduction = self.encoder.reduction_factors[-1]
            h = original_input_x.shape[2] // final_reduction
            w = original_input_x.shape[3] // final_reduction
        else:
            logger.warning(
                "Could not reliably determine H, W for (B,L,C) "
                "bottleneck reshape. Estimating based on L."
            )
            side = int(L**0.5)
            if side * side == L:
                h, w = side, side
        return h, w

    def _reshape_3d_bottleneck(
        self, bottleneck_features: torch.Tensor, original_input_x: torch.Tensor
    ) -> torch.Tensor:
        """Handles reshaping for 3D bottleneck features."""
        B, L, C_actual = bottleneck_features.shape
        expected_channels = self.encoder.out_channels
        if C_actual != expected_channels:
            logger.warning(
                f"Bottleneck features (B,L,C) have "
                f"C={C_actual} but expected {expected_channels}"
            )

        h, w = self._determine_hw_for_3d_reshape(L, original_input_x)

        if h > 0 and w > 0 and L == h * w:
            bottleneck_features = (
                bottleneck_features.view(B, h, w, C_actual)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            logger.debug(
                "Reshaped and Permuted bottleneck features to "
                f"{bottleneck_features.shape}"
            )
        elif L > 0 and C_actual > 0:
            logger.warning(
                "Could not reshape (B,L,C) bottleneck "
                f"features {bottleneck_features.shape} to "
                "spatial format as h,w were not inferred correctly."
                "Proceeding with (B,L,C) format if downstream handles it."
            )
        else:
            logger.error(
                "Could not reshape (B,L,C) bottleneck features "
                f"{bottleneck_features.shape} to spatial format."
            )
            raise ValueError(
                "Failed to determine spatial dims for "
                "bottleneck reshape from (B,L,C)"
            )
        return bottleneck_features

    def _reshape_bottleneck_if_needed(
        self, bottleneck_features: torch.Tensor, original_input_x: torch.Tensor
    ) -> torch.Tensor:
        """Reshapes bottleneck features to (B, C, H, W) if necessary."""
        if bottleneck_features.ndim == self.expected_bottleneck_ndim_4d:
            bottleneck_features = self._reshape_4d_bottleneck(
                bottleneck_features
            )
        elif bottleneck_features.ndim == self.expected_bottleneck_ndim_3d:
            bottleneck_features = self._reshape_3d_bottleneck(
                bottleneck_features, original_input_x
            )
        else:
            logger.warning(
                "Bottleneck features have unexpected ndim: "
                f"{bottleneck_features.ndim}"
            )
        return bottleneck_features

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the SwinV2 encoder adapter.

        Calls the forward method of the underlying SwinTransformerEncoder,
        and ensures the bottleneck output is in (B, C, H, W) format.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final encoder output (bottleneck features) in (B, C, H, W).
                - List of intermediate feature maps (skip connections)
                  ordered from higher to lower resolution.
        """
        # The forward method of SwinTransformerEncoder returns:
        # (bottleneck, [skip1, skip2, ...]) H->L res
        bottleneck_features, skip_connections = self.encoder.forward(x)

        bottleneck_features = self._reshape_bottleneck_if_needed(
            bottleneck_features, x
        )

        # Return the potentially permuted/reshaped bottleneck and original
        # skips
        return bottleneck_features, skip_connections

    def get_optimizer_param_groups(
        self, base_lr: float = 0.001
    ) -> list[dict[str, Any]]:
        """
        Returns parameter groups for differential learning rates.
        Delegates to the underlying encoder's method if available.
        """
        if hasattr(self.encoder, "get_optimizer_param_groups"):
            return self.encoder.get_optimizer_param_groups(base_lr)
        else:
            # Fallback if the underlying encoder doesn't support it
            return [{"params": self.parameters(), "lr": base_lr}]

    def __str__(self) -> str:
        return f"SwinV2EncoderAdapter(Wrapped: {str(self.encoder)})"

    def __repr__(self) -> str:
        return self.__str__()


# TODO: Add registration logic if needed by a factory
# Example:
# from src.model.encoder_registry import register_encoder
# register_encoder("swin_v2_adapter", SwinV2EncoderAdapter)
