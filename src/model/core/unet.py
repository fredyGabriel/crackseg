"""
Base UNet model implementation that integrates abstract components.

This module provides a concrete implementation of the UNetBase abstract
class that integrates EncoderBase, BottleneckBase, and DecoderBase
components into a complete UNet model.
"""

from io import StringIO
from typing import Any, cast

import hydra.utils  # Import hydra utils
import torch
from torch import nn

# Actualizar importaciones para reflejar la nueva estructura de directorios
from src.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from src.model.common.utils import (
    count_parameters,
    estimate_memory_usage,
    estimate_receptive_field,
    get_layer_hierarchy,
    render_unet_architecture_diagram,
)


class BaseUNet(UNetBase):
    """
    Base UNet model implementation that integrates abstract components.

    This class provides a concrete implementation of the UNetBase abstract
    class, combining encoder, bottleneck and decoder components into a
    complete U-Net architecture for image segmentation tasks.
    """

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
        final_activation: nn.Module | dict[str, Any] | None = None,
    ):
        """
        Initialize the BaseUNet model.

        Args:
            encoder (EncoderBase): Encoder component for feature extraction.
            bottleneck (BottleneckBase): Bottleneck component for feature
                processing at the lowest resolution.
            decoder (DecoderBase): Decoder component for upsampling and
                generating the segmentation map.
            final_activation (Optional[Union[nn.Module, Dict[str, Any]]]):
                Optional activation function, either as an nn.Module instance
                or a Hydra configuration dict. Default: None.
        """
        super().__init__(encoder, bottleneck, decoder)

        # Instantiate final activation
        self.final_activation = None
        if final_activation is not None:
            # If it's already an nn.Module instance, use it directly
            if isinstance(final_activation, nn.Module):
                self.final_activation = final_activation
            # Otherwise, try to instantiate from config
            else:
                try:
                    self.final_activation = hydra.utils.instantiate(
                        final_activation
                    )
                    if not isinstance(self.final_activation, nn.Module):
                        raise TypeError(
                            "final_activation config did not instantiate an \
nn.Module"
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not instantiate final_activation: \
{e}"
                    )
                    self.final_activation = None  # Fallback to no activation

        # Validar compatibilidad de canales de skip
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.bottleneck is not None
        if hasattr(self.encoder, "skip_channels") and hasattr(
            self.decoder, "skip_channels"
        ):
            if list(self.encoder.skip_channels) != list(
                reversed(self.decoder.skip_channels)
            ):
                raise ValueError(
                    "Encoder skip channels and decoder skip channels are "
                    "incompatible."
                )
        # Validar compatibilidad de canales entre encoder y bottleneck
        if hasattr(self.encoder, "out_channels") and hasattr(
            self.bottleneck, "in_channels"
        ):
            if self.encoder.out_channels != self.bottleneck.in_channels:
                raise ValueError(
                    "Encoder output channels and bottleneck input channels "
                    "are incompatible."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                B: batch size, C: input channels, H: height, W: width.

        Returns:
            torch.Tensor: Output segmentation map of shape [B, O, H, W],
                where O is the number of output channels.
        """
        assert self.encoder is not None
        assert self.bottleneck is not None
        assert self.decoder is not None
        # Pass input through encoder to get features and skip connections
        features, skip_connections = self.encoder(x)

        # Pass features through bottleneck
        bottleneck_output = self.bottleneck(features)

        # Pass bottleneck output and skip connections through decoder
        output = self.decoder(bottleneck_output, skip_connections)

        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)

        return cast(torch.Tensor, output)

    def get_output_channels(self) -> int:
        """
        Get the number of output channels from the model.

        Returns:
            int: Number of output channels (from decoder).
        """
        assert self.decoder is not None
        return self.decoder.out_channels

    def get_input_channels(self) -> int:
        """
        Get the number of input channels the model expects.

        Returns:
            int: Number of input channels (from encoder).
        """
        assert self.encoder is not None
        return self.encoder.in_channels

    def summary(
        self, input_shape: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        """
        Generate a complete summary of the model architecture.

        Args:
            input_shape (Tuple[int, ...]): Optional input shape (B, C, H, W)
                for memory estimation.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information.
        """
        trainable_params, non_trainable_params = count_parameters(self)
        total_params = trainable_params + non_trainable_params
        base_info = {
            "model_type": self.__class__.__name__,
            "input_channels": self.get_input_channels(),
            "output_channels": self.get_output_channels(),
            "encoder_type": self.encoder.__class__.__name__,
            "bottleneck_type": self.bottleneck.__class__.__name__,
            "decoder_type": self.decoder.__class__.__name__,
            "has_final_activation": self.final_activation is not None,
            "final_activation_type": (
                self.final_activation.__class__.__name__
                if self.final_activation is not None
                else None
            ),
        }
        summary_dict = {
            **base_info,
            "parameters": {
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": non_trainable_params,
                "trainable_percent": (
                    trainable_params / total_params * 100
                    if total_params > 0
                    else 0
                ),
            },
            "receptive_field": estimate_receptive_field(self.encoder),
            "memory_usage": estimate_memory_usage(
                self, self.encoder, self.get_output_channels, input_shape
            ),
            "layer_hierarchy": get_layer_hierarchy(
                self.encoder,
                self.bottleneck,
                self.decoder,
                self.final_activation,
            ),
        }
        return summary_dict

    def _print_header_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        print("\n" + "=" * 80, file=target_file)
        print("U-Net Model Summary", file=target_file)
        print("=" * 80, file=target_file)
        print(f"\nModel Type: {summary_dict['model_type']}", file=target_file)
        print(
            f"Input Channels: {summary_dict['input_channels']}",
            file=target_file,
        )
        print(
            f"Output Channels: {summary_dict['output_channels']}",
            file=target_file,
        )

    def _print_parameter_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        params = summary_dict["parameters"]
        print(f"\nTotal Parameters: {params['total']:,}", file=target_file)
        print(
            f"Trainable Parameters: {params['trainable']:,} "
            f"({params['trainable_percent']:.2f}%)",
            file=target_file,
        )
        print(
            f"Non-trainable Parameters: {params['non_trainable']:,}",
            file=target_file,
        )

    def _print_memory_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        mem = summary_dict["memory_usage"]
        print(f"\nModel Size: {mem['model_size_mb']:.2f} MB", file=target_file)
        if "estimated_activation_mb" in mem:
            print(
                f"Estimated Activation Memory: "
                f"{mem['estimated_activation_mb']:.2f} MB",
                file=target_file,
            )
            print(
                f"Total Estimated Memory: {mem['total_estimated_mb']:.2f} MB",
                file=target_file,
            )
            print(f"For input shape: {mem['input_shape']}", file=target_file)

    def _print_receptive_field_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        rf = summary_dict["receptive_field"]
        print("\nReceptive Field:", file=target_file)
        if "theoretical_rf_size" in rf:
            print(
                f"  Theoretical Size: {rf['theoretical_rf_size']}x"
                f"{rf['theoretical_rf_size']}",
                file=target_file,
            )
            print(
                f"  Downsampling Factor: {rf['downsampling_factor']}",
                file=target_file,
            )
        if "note" in rf:
            print(f"  Note: {rf['note']}", file=target_file)

    def _print_architecture_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        print("\nArchitecture:", file=target_file)
        print(f"  Encoder: {summary_dict['encoder_type']}", file=target_file)
        print(
            f"  Bottleneck: {summary_dict['bottleneck_type']}",
            file=target_file,
        )
        print(f"  Decoder: {summary_dict['decoder_type']}", file=target_file)
        if summary_dict["has_final_activation"]:
            print(
                f" Final Activation: {summary_dict['final_activation_type']}",
                file=target_file,
            )

    def _print_layer_hierarchy_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        print("\nLayer Hierarchy:", file=target_file)
        print("-" * 80, file=target_file)
        print(
            f"{'Layer':<25} {'Type':<20} {'Parameters':>12} "
            f"{'Input Ch':>10} {'Output Ch':>10}",
            file=target_file,
        )
        print("-" * 80, file=target_file)
        for layer in summary_dict["layer_hierarchy"]:
            layer_name = layer["name"]
            layer_type = layer.get("type", "")
            params = layer.get("params", 0)
            in_ch = layer.get("in_channels", "")
            out_ch = layer.get("out_channels", "")
            print(
                f"{layer_name:<25} {layer_type:<20} {params:>12,} "
                f"{in_ch:>10} {out_ch:>10}",
                file=target_file,
            )
            if "blocks" in layer:
                for block in layer["blocks"]:
                    block_name = "  " + block["name"]
                    block_type = block.get("type", "")
                    block_params = block.get("params", 0)
                    block_in_ch = block.get("in_channels", "")
                    block_out_ch = block.get("out_channels", "")
                    print(
                        f"{block_name:<25} {block_type:<20} "
                        f"{block_params:>12,} {block_in_ch:>10} "
                        f"{block_out_ch:>10}",
                        file=target_file,
                    )
        print("-" * 80, file=target_file)
        print("=" * 80 + "\n", file=target_file)

    def print_summary(
        self,
        input_shape: tuple[int, ...] | None = None,
        file: Any | None = None,
        return_string: bool = False,
    ) -> str | None:
        """
        Print a formatted summary of the model architecture.

        Args:
            input_shape (Tuple[int, ...]): Optional input shape (B, C, H, W)
                for memory estimation.
            file (Optional[Any]): File object to write the summary to.
                If None, writes to sys.stdout.
            return_string (bool): Whether to return the summary as a string.
                                 Default: False

        Returns:
            Optional[str]: Formatted summary string if return_string is True,
                          otherwise None.
        """
        string_stream = StringIO()  # Always initialize
        target_file: Any
        if return_string:
            target_file = string_stream
        else:
            import sys

            target_file = file if file is not None else sys.stdout

        summary_dict = self.summary(input_shape)

        self._print_header_info(summary_dict, target_file)
        self._print_parameter_info(summary_dict, target_file)
        self._print_memory_info(summary_dict, target_file)
        self._print_receptive_field_info(summary_dict, target_file)
        self._print_architecture_info(summary_dict, target_file)
        self._print_layer_hierarchy_info(summary_dict, target_file)

        if return_string:
            return string_stream.getvalue()
        return None

    def visualize_architecture(
        self, filename: str | None = None, view: bool = False
    ) -> None:
        """
        Generate a U-shaped block diagram of the U-Net architecture using
        graphviz.
        """
        layer_hierarchy = get_layer_hierarchy(
            self.encoder, self.bottleneck, self.decoder, self.final_activation
        )
        render_unet_architecture_diagram(layer_hierarchy, filename, view)
