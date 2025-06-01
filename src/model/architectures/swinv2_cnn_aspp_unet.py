# src/model/architectures/swinv2_cnn_aspp_unet.py

import logging
from typing import Any, cast

import torch
from torch import nn

# Base class
from src.model.base.abstract import BottleneckBase, DecoderBase, UNetBase
from src.model.components.aspp import ASPPModule
from src.model.decoder.cnn_decoder import CNNDecoder

# Components
from src.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter

# Activation factory if needed
# from src.model.factory import create_activation # Assuming exists

logger = logging.getLogger(__name__)


class SwinV2CnnAsppUNet(UNetBase):
    """
    Hybrid U-Net Architecture combining SwinV2 Encoder, ASPP Bottleneck,
    and CNN Decoder with optional CBAM attention.

    Inherits from UNetBase and orchestrates the data flow between the
    specialized encoder, bottleneck, and decoder components.
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
        """
        Initializes the SwinV2CnnAsppUNet model.

        Args:
            encoder_cfg (Dict[str, Any]): Configuration dictionary for the
                SwinV2EncoderAdapter.
            bottleneck_cfg (Dict[str, Any]): Configuration dictionary for the
                ASPPModule bottleneck.
            decoder_cfg (Dict[str, Any]): Configuration dictionary for the
                CNNDecoder.
            num_classes (int): Number of output segmentation classes.
                               Default: 1 (binary segmentation).
            final_activation (Optional[str]): Name of the final activation
                function ('sigmoid', 'softmax', etc.) or None.
                Default: 'sigmoid'.
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
        """
        Defines the forward pass for the complete U-Net model.

        Args:
            x (torch.Tensor): Input tensor (batch of images) with shape
                             (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor (segmentation map/logits) with shape
                         (batch_size, num_classes, height, width).
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
# from src.model.model_registry import register_model
# register_model("swinv2_cnn_aspp_unet", SwinV2CnnAsppUNet)
