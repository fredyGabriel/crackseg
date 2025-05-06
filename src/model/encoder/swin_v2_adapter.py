# src/model/encoder/swin_v2_adapter.py
import torch
from typing import List, Any, Optional, Tuple, Dict, Union
import logging

# Corrected imports based on file search and class definitions
from src.model.base import EncoderBase
from src.model.encoder.swin_transformer_encoder import SwinTransformerEncoder

logger = logging.getLogger(__name__)


class SwinV2EncoderAdapter(EncoderBase):
    """
    Adapter class for the Swin Transformer V2 encoder (from Task 17)
    to integrate with the U-Net architecture based on EncoderBase.

    This adapter wraps the SwinTransformerEncoder implementation and ensures
    it conforms to the EncoderBase interface.
    """

    def __init__(
        self,
        # Parameters passed directly to SwinTransformerEncoder
        in_channels: int,
        model_name: str = "swinv2_tiny_window16_256",
        pretrained: bool = True,
        out_indices: Optional[List[int]] = None,
        img_size: int = 256,
        patch_size: int = 4,
        handle_input_size: str = "resize",
        freeze_layers: Union[bool, str, List[str]] = False,
        finetune_lr_scale: Dict[str, float] = None,
        # Kwargs for potential EncoderBase needs or future flexibility
        **kwargs: Any
    ):
        """
        Initializes the SwinV2EncoderAdapter.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            model_name (str): Name of the Swin Transformer V2 model from timm.
                              Default: 'swinv2_tiny_window16_256'.
            pretrained (bool): Whether to use pretrained weights. Default: True
            out_indices (List[int], optional): Indices of the feature maps to
                                              return. Default: uses encoder's
                                              default [0, 1, 2, 3].
            img_size (int): Input image size for the model. Default: 256.
            patch_size (int): Patch size. Default: 4.
            handle_input_size (str): How to handle inputs with sizes different
                                    from img_size ('resize', 'pad', 'none').
                                    Default: 'resize'.
            freeze_layers (Union[bool, str, List[str]]): Control layer freezing
                                    for transfer learning. Default: False.
            finetune_lr_scale (Dict[str, float]): Optional LR scales for layers
                                    during fine-tuning. Default: None.
            **kwargs: Additional keyword arguments (currently unused).
        """
        # Pass in_channels to EncoderBase constructor
        super().__init__(in_channels)

        # Instantiate the actual SwinTransformerEncoder from Task 17
        # Pass only the relevant parameters
        self.encoder = SwinTransformerEncoder(
            in_channels=in_channels,
            model_name=model_name,
            pretrained=pretrained,
            output_hidden_states=True,  # Required to get skips
            features_only=True,       # Required for feature maps
            out_indices=out_indices,  # Use provided or encoder default
            img_size=img_size,
            patch_size=patch_size,
            use_abs_pos_embed=True,   # Assuming default from SwinTEnc
            output_norm=True,         # Assuming default from SwinTEnc
            handle_input_size=handle_input_size,
            freeze_layers=freeze_layers,
            finetune_lr_scale=finetune_lr_scale,
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
    def skip_channels(self) -> List[int]:
        """List of channels for skip connections (high to low resolution)."""
        # Directly use the property from the wrapped encoder
        return self.encoder.skip_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
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

        # *** FIX: Ensure bottleneck features are (B, C, H, W) ***
        # Check dimensions. If 4 dims and channels last, permute.
        if bottleneck_features.ndim == 4:
            # Example shape might be (B, H, W, C) from some timm backbones
            # Get inferred channels from the property
            expected_channels = self.encoder.out_channels
            if bottleneck_features.shape[1] == expected_channels:
                # Shape is likely (B, C, H, W) - Correct format
                pass
            elif bottleneck_features.shape[3] == expected_channels:
                # Shape is likely (B, H, W, C) - Permute to (B, C, H, W)
                bottleneck_features = bottleneck_features.permute(
                    0, 3, 1, 2).contiguous()
                logger.debug("Permuted bottleneck features to "
                             f"{bottleneck_features.shape}")
            else:
                # Shape is unexpected, log warning
                logger.warning(
                    f"Bottleneck features have unexpected shape: "
                    f"{bottleneck_features.shape}. Expected channels "
                    f"{expected_channels} at dim 1 or 3."
                )
        elif bottleneck_features.ndim == 3:
            # Output might be (B, L, C) -> Need to reshape AND permute
            B, L, C_actual = bottleneck_features.shape
            expected_channels = self.encoder.out_channels
            if C_actual != expected_channels:
                logger.warning(f"Bottleneck features (B,L,C) have "
                               f"C={C_actual} but expected {expected_channels}"
                               )
            # Try to infer H, W (Requires encoder to provide info)
            H, W = -1, -1
            if hasattr(self.encoder.encoder, 'get_feature_info'):
                final_stage_info = self.encoder.encoder.get_feature_info()[-1]
                reduction_factor = final_stage_info.get('reduction_factor')
                if reduction_factor:
                    H = x.shape[2] // reduction_factor
                    W = x.shape[3] // reduction_factor
            if H > 0 and W > 0 and L == H * W:
                bottleneck_features = bottleneck_features.view(
                    B, H, W, C_actual).permute(0, 3, 1, 2).contiguous()
                logger.debug("Reshaped and Permuted bottleneck features to "
                             f"{bottleneck_features.shape}")
            else:
                logger.error("Could not reshape (B,L,C) bottleneck features "
                             f"{bottleneck_features.shape} to spatial format.")
                # Raise error or handle differently?
                raise ValueError("Failed to determine spatial dims for "
                                 "bottleneck reshape from (B,L,C)")
        else:
            logger.warning("Bottleneck features have unexpected ndim: "
                           f"{bottleneck_features.ndim}")

        # Return the potentially permuted/reshaped bottleneck and original
        # skips
        return bottleneck_features, skip_connections

    def get_optimizer_param_groups(self, base_lr: float = 0.001) -> List[Dict]:
        """
        Returns parameter groups for differential learning rates.
        Delegates to the underlying encoder's method if available.
        """
        if hasattr(self.encoder, 'get_optimizer_param_groups'):
            return self.encoder.get_optimizer_param_groups(base_lr)
        else:
            # Fallback if the underlying encoder doesn't support it
            return [{'params': self.parameters(), 'lr': base_lr}]

    def __str__(self) -> str:
        return f"SwinV2EncoderAdapter(Wrapped: {str(self.encoder)})"

    def __repr__(self) -> str:
        return self.__str__()

# TODO: Add registration logic if needed by a factory
# Example:
# from src.model.encoder_registry import register_encoder
# register_encoder("swin_v2_adapter", SwinV2EncoderAdapter)
