"""
Base UNet model implementation that integrates abstract components.

This module provides a concrete implementation of the UNetBase abstract
class that integrates EncoderBase, BottleneckBase, and DecoderBase
components into a complete UNet model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import hydra.utils  # Import hydra utils
import sys
from io import StringIO

from src.model.base import UNetBase, EncoderBase, BottleneckBase, DecoderBase


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
        final_activation: Optional[Union[nn.Module, Dict[str, Any]]] = None
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
        self.final_activation: Optional[nn.Module] = None
        if final_activation is not None:
            # If it's already an nn.Module instance, use it directly
            if isinstance(final_activation, nn.Module):
                self.final_activation = final_activation
            # Otherwise, try to instantiate from config
            else:
                try:
                    self.final_activation = hydra.utils.instantiate(
                        final_activation)
                    if not isinstance(self.final_activation, nn.Module):
                        raise TypeError(
                            "final_activation config did not instantiate an \
nn.Module"
                        )
                except Exception as e:
                    print(f"Warning: Could not instantiate final_activation: \
{e}")
                    self.final_activation = None  # Fallback to no activation

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
        # Pass input through encoder to get features and skip connections
        features, skip_connections = self.encoder(x)

        # Pass features through bottleneck
        bottleneck_output = self.bottleneck(features)

        # Pass bottleneck output and skip connections through decoder
        output = self.decoder(bottleneck_output, skip_connections)

        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)

        return output

    def get_output_channels(self) -> int:
        """
        Get the number of output channels from the model.

        Returns:
            int: Number of output channels (from decoder).
        """
        return self.decoder.out_channels

    def get_input_channels(self) -> int:
        """
        Get the number of input channels the model expects.

        Returns:
            int: Number of input channels (from encoder).
        """
        return self.encoder.in_channels

    def _count_parameters(self) -> Tuple[int, int]:
        """
        Count the trainable and non-trainable parameters in the model.

        Returns:
            Tuple[int, int]: (trainable_params, non_trainable_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if
                        p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not
                            p.requires_grad)
        return trainable, non_trainable

    def _estimate_receptive_field(self) -> Dict[str, Any]:
        """
        Estimate the receptive field size of the model.

        For U-Net, the theoretical receptive field depends on the depth
        and kernel sizes. This is an approximation for standard CNN U-Net
        with 3x3 kernels and 2x2 pooling.

        Returns:
            Dict[str, Any]: Receptive field information
        """
        # Try to determine depth from encoder if it's a CNNEncoder
        depth = getattr(self.encoder, 'depth', None)

        if depth is not None:
            # For standard U-Net with 3x3 kernels and depth levels of pooling
            # Each encoder level adds 4 to the receptive field (2 conv layers)
            # and each decoder level also adds 4
            # Then we double for each level of pooling
            receptive_field_size = 3 + (depth * 4 * 2) - 1
            downsampling_factor = 2 ** depth
            return {
                "theoretical_rf_size": receptive_field_size,
                "downsampling_factor": downsampling_factor,
                "note": "Theoretical estimate for standard U-Net with 3x3 \
kernels"
            }
        else:
            # If depth is not available, provide a generic message
            return {
                "note": "Receptive field estimation requires a standard \
encoder with known depth"
            }

    def _estimate_memory_usage(self, input_shape: Tuple[int, ...] = None
                               ) -> Dict[str, Any]:
        """
        Estimate memory usage for the model.

        Args:
            input_shape (Tuple[int, ...]): Optional input shape (B, C, H, W).
                If not provided, a reasonable default is used.

        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        param_bytes = sum(p.numel() * p.element_size()
                          for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size()
                           for b in self.buffers())

        model_size_mb = (param_bytes + buffer_bytes) / (1024 * 1024)

        # If input shape is provided, estimate activation memory
        if input_shape:
            B, C, H, W = input_shape
            # Rough estimate: assume memory at each layer scales with spatial
            # dimensions
            # This is a simplification and actual usage depends on
            # architecture details
            # Factor in depth if available
            # Default to 4 if not available
            depth = getattr(self.encoder, 'depth', 4)

            # Encoder: each level has feature maps at decreasing resolution
            encoder_memory = 0
            for i in range(depth):
                # Common U-Net feature pattern
                features = min(64 * (2 ** i), 512)
                h, w = H // (2 ** i), W // (2 ** i)
                # 4 bytes per float32
                encoder_memory += B * features * h * w * 4

            # Bottleneck
            bottleneck_features = min(64 * (2 ** depth), 1024)
            bottleneck_h = H // (2 ** depth)
            bottleneck_w = W // (2 ** depth)
            bottleneck_memory = B * bottleneck_features * bottleneck_h
            bottleneck_memory *= bottleneck_w * 4

            # Decoder: each level has feature maps at increasing resolution
            decoder_memory = 0
            for i in range(depth):
                j = depth - i - 1
                features = min(64 * (2 ** j), 512)
                h, w = H // (2 ** j), W // (2 ** j)
                decoder_memory += B * features * h * w * 4

            # Final output
            output_memory = B * self.get_output_channels() * H * W * 4

            activation_mb = (encoder_memory + bottleneck_memory +
                             decoder_memory + output_memory) / (1024 * 1024)

            return {
                "model_size_mb": model_size_mb,
                "estimated_activation_mb": activation_mb,
                "total_estimated_mb": model_size_mb + activation_mb,
                "input_shape": input_shape
            }

        return {
            "model_size_mb": model_size_mb,
            "note": "For activation memory estimates, provide input_shape"
        }

    def _get_layer_hierarchy(self) -> List[Dict[str, Any]]:
        """
        Get the hierarchical structure of the model layers.

        Returns:
            List[Dict[str, Any]]: List of layer information dictionaries
        """
        hierarchy = []

        # Encoder section
        encoder_info = {
            "name": "Encoder",
            "type": self.encoder.__class__.__name__,
            "params": sum(p.numel() for p in self.encoder.parameters()),
            "out_channels": self.encoder.out_channels,
            "skip_channels": self.encoder.skip_channels
        }

        # Add details for CNNEncoder
        if hasattr(self.encoder, 'encoder_blocks'):
            blocks_info = []
            for i, block in enumerate(self.encoder.encoder_blocks):
                block_info = {
                    "name": f"EncoderBlock_{i+1}",
                    "params": sum(p.numel() for p in block.parameters()),
                    "in_channels": block.in_channels,
                    "out_channels": block.out_channels
                }
                blocks_info.append(block_info)
            encoder_info["blocks"] = blocks_info

        hierarchy.append(encoder_info)

        # Bottleneck section
        bottleneck_info = {
            "name": "Bottleneck",
            "type": self.bottleneck.__class__.__name__,
            "params": sum(p.numel() for p in self.bottleneck.parameters()),
            "in_channels": self.bottleneck.in_channels,
            "out_channels": self.bottleneck.out_channels
        }
        hierarchy.append(bottleneck_info)

        # Decoder section
        decoder_info = {
            "name": "Decoder",
            "type": self.decoder.__class__.__name__,
            "params": sum(p.numel() for p in self.decoder.parameters()),
            "in_channels": self.decoder.in_channels,
            "out_channels": self.decoder.out_channels,
            "skip_channels": self.decoder.skip_channels
        }

        # Add details for CNNDecoder
        if hasattr(self.decoder, 'decoder_blocks'):
            blocks_info = []
            for i, block in enumerate(self.decoder.decoder_blocks):
                block_info = {
                    "name": f"DecoderBlock_{i+1}",
                    "params": sum(p.numel() for p in block.parameters()),
                    "in_channels": block.in_channels,
                    "out_channels": block.out_channels
                }
                blocks_info.append(block_info)

            # Add final conv if available
            if hasattr(self.decoder, 'final_conv'):
                final_conv_info = {
                    "name": "FinalConv",
                    "params": sum(p.numel() for p in
                                  self.decoder.final_conv.parameters()),
                    "in_channels": self.decoder.final_conv.in_channels,
                    "out_channels": self.decoder.final_conv.out_channels
                }
                blocks_info.append(final_conv_info)

            decoder_info["blocks"] = blocks_info

        hierarchy.append(decoder_info)

        # Final activation if present
        if self.final_activation is not None:
            activation_info = {
                "name": "FinalActivation",
                "type": self.final_activation.__class__.__name__,
                "params": sum(p.numel() for p in
                              self.final_activation.parameters())
            }
            hierarchy.append(activation_info)

        return hierarchy

    def summary(self, input_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
        """
        Generate a complete summary of the model architecture.

        Args:
            input_shape (Tuple[int, ...]): Optional input shape (B, C, H, W)
                for memory estimation.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information.
        """
        # Count parameters
        trainable_params, non_trainable_params = self._count_parameters()
        total_params = trainable_params + non_trainable_params

        # Get basic model information
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
                if self.final_activation is not None else None
            )
        }

        # Combine all information
        summary_dict = {
            **base_info,
            "parameters": {
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": non_trainable_params,
                "trainable_percent": trainable_params / total_params * 100 if
                total_params > 0 else 0
            },
            "receptive_field": self._estimate_receptive_field(),
            "memory_usage": self._estimate_memory_usage(input_shape),
            "layer_hierarchy": self._get_layer_hierarchy()
        }

        return summary_dict

    def print_summary(self, input_shape: Tuple[int, ...] = None,
                      file: Optional[Any] = None,
                      return_string: bool = False) -> Optional[str]:
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
        if return_string:
            string_stream = StringIO()
            target_file = string_stream
        else:
            target_file = file or sys.stdout

        # Get the full summary dict
        summary_dict = self.summary(input_shape)

        # Format the summary
        print("\n" + "=" * 80, file=target_file)
        print("U-Net Model Summary", file=target_file)
        print("=" * 80, file=target_file)

        # Model overview
        print(f"\nModel Type: {summary_dict['model_type']}", file=target_file)
        print(f"Input Channels: {summary_dict['input_channels']}",
              file=target_file)
        print(f"Output Channels: {summary_dict['output_channels']}",
              file=target_file)

        # Parameter counts
        params = summary_dict["parameters"]
        print(f"\nTotal Parameters: {params['total']:,}", file=target_file)
        print(f"Trainable Parameters: {params['trainable']:,} \
({params['trainable_percent']:.2f}%)", file=target_file)
        print(f"Non-trainable Parameters: {params['non_trainable']:,}",
              file=target_file)

        # Memory usage
        mem = summary_dict["memory_usage"]
        print(f"\nModel Size: {mem['model_size_mb']:.2f} MB", file=target_file)
        if "estimated_activation_mb" in mem:
            print(f"Estimated Activation Memory: \
{mem['estimated_activation_mb']:.2f} MB", file=target_file)
            print(f"Total Estimated Memory: \
{mem['total_estimated_mb']:.2f} MB", file=target_file)
            print(f"For input shape: {mem['input_shape']}", file=target_file)

        # Receptive field
        rf = summary_dict["receptive_field"]
        print("\nReceptive Field:", file=target_file)
        if "theoretical_rf_size" in rf:
            print(f"  Theoretical Size: {rf['theoretical_rf_size']}x\
{rf['theoretical_rf_size']}", file=target_file)
            print(f"  Downsampling Factor: {rf['downsampling_factor']}",
                  file=target_file)
        if "note" in rf:
            print(f"  Note: {rf['note']}", file=target_file)

        # Architecture components
        print("\nArchitecture:", file=target_file)
        print(f"  Encoder: {summary_dict['encoder_type']}", file=target_file)
        print(f"  Bottleneck: {summary_dict['bottleneck_type']}",
              file=target_file)
        print(f"  Decoder: {summary_dict['decoder_type']}", file=target_file)
        if summary_dict['has_final_activation']:
            print(f" Final Activation: \
{summary_dict['final_activation_type']}", file=target_file)

        # Layer hierarchy
        print("\nLayer Hierarchy:", file=target_file)
        print("-" * 80, file=target_file)
        print(f"{'Layer':<25} {'Type':<20} {'Parameters':>12} \
{'Input Ch':>10} {'Output Ch':>10}", file=target_file)
        print("-" * 80, file=target_file)

        for layer in summary_dict["layer_hierarchy"]:
            layer_name = layer["name"]
            layer_type = layer.get("type", "")
            params = layer.get("params", 0)
            in_ch = layer.get("in_channels", "")
            out_ch = layer.get("out_channels", "")

            print(f"{layer_name:<25} {layer_type:<20} {params:>12,} \
{in_ch:>10} {out_ch:>10}", file=target_file)

            # Print sub-blocks if available
            if "blocks" in layer:
                for block in layer["blocks"]:
                    block_name = "  " + block["name"]
                    block_type = block.get("type", "")
                    block_params = block.get("params", 0)
                    block_in_ch = block.get("in_channels", "")
                    block_out_ch = block.get("out_channels", "")

                    print(f"{block_name:<25} {block_type:<20} \
{block_params:>12,} {block_in_ch:>10} {block_out_ch:>10}", file=target_file)

        print("-" * 80, file=target_file)
        print("=" * 80 + "\n", file=target_file)

        if return_string:
            return string_stream.getvalue()
        return None

    def visualize_architecture(self, filename: str = None, view: bool = False
                               ) -> None:
        """
        Generate a U-shaped block diagram of the U-Net architecture using
        graphviz: encoder descendente (izquierda), bottleneck en la base,
        decoder ascendente (derecha).

        Args:
            filename (str): If provided, saves the diagram to this file (e.g.,
            'unet_architecture.gv' or 'unet_architecture.png').
            view (bool): If True, opens the diagram after creation (requires a
            supported viewer).
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError(
                "graphviz is required for visualize_architecture. "
                "Install with 'conda install graphviz python-graphviz'."
            )

        # Crear un grafo con formato estricto para mejor layout
        dot = Digraph(comment="U-Net Architecture", format="png", strict=True)
        # Usar orientación vertical y agregar atributos para mejor
        # visualización
        dot.attr(rankdir="TB", splines="ortho", nodesep="0.5",
                 ranksep="0.5", fontsize="12")

        # Obtener información de bloques
        encoder_blocks = getattr(self.encoder, "encoder_blocks", None)
        decoder_blocks = getattr(self.decoder, "decoder_blocks", None)

        # Determinar la profundidad (número de niveles)
        if encoder_blocks:
            depth = len(encoder_blocks)
        else:
            depth = 1

        # Crear nodo de entrada
        dot.node("Input", "Input\n[Batch, C, H, W]", shape="oval",
                 style="filled", fillcolor="#e0e0e0")

        # --- COLUMNA IZQUIERDA: ENCODER (DESCENDENTE) ---
        encoder_nodes = []
        with dot.subgraph(name='cluster_encoder') as c:
            c.attr(rank='same')  # Misma columna
            if encoder_blocks:
                for i, block in enumerate(encoder_blocks):
                    node_name = f"Enc{i+1}"
                    c.node(
                        node_name,
                        f"EncoderBlock {i+1}\n({block.in_channels}→\
 {block.out_channels})",
                        shape="box", style="filled", fillcolor="#b3cde0"
                    )
                    encoder_nodes.append(node_name)
            else:
                c.node(
                    "Encoder", f"{self.encoder.__class__.__name__}",
                    shape="box", style="filled", fillcolor="#b3cde0"
                )
                encoder_nodes.append("Encoder")

        # --- BASE: BOTTLENECK ---
        dot.node(
            "Bottleneck",
            f"Bottleneck\n({self.bottleneck.in_channels}→\
 {self.bottleneck.out_channels})",
            shape="box", style="filled", fillcolor="#fbb4ae"
        )

        # --- COLUMNA DERECHA: DECODER (ASCENDENTE) ---
        decoder_nodes = []
        with dot.subgraph(name='cluster_decoder') as c:
            c.attr(rank='same')  # Misma columna
            if decoder_blocks:
                for i, block in enumerate(decoder_blocks):
                    node_name = f"Dec{i+1}"
                    c.node(
                        node_name,
                        f"DecoderBlock {i+1}\n({block.in_channels}→\
 {block.out_channels})",
                        shape="box", style="filled", fillcolor="#ccebc5"
                    )
                    decoder_nodes.append(node_name)
            else:
                c.node(
                    "Decoder", f"{self.decoder.__class__.__name__}",
                    shape="box", style="filled", fillcolor="#ccebc5"
                )
                decoder_nodes.append("Decoder")

        # --- SALIDA ---
        final_nodes = []
        if hasattr(self.decoder, "final_conv"):
            fc = self.decoder.final_conv
            dot.node(
                "FinalConv",
                f"FinalConv\n({fc.in_channels}→{fc.out_channels})",
                shape="box", style="filled", fillcolor="#decbe4"
            )
            final_nodes.append("FinalConv")

        if self.final_activation is not None:
            dot.node(
                "Activation",
                f"{self.final_activation.__class__.__name__}",
                shape="ellipse", style="filled", fillcolor="#fed9a6"
            )
            final_nodes.append("Activation")

        dot.node(
            "Output", "Output\n[Batch, C, H, W]", shape="oval",
            style="filled", fillcolor="#e0e0e0"
        )
        final_nodes.append("Output")

        # --- POSICIONAMIENTO DE COLUMNAS ---
        for i in range(depth):
            with dot.subgraph() as s:
                s.attr(rank=f'level{i}')  # Nivel en el grafo
                if i < len(encoder_nodes):
                    s.node(encoder_nodes[i])
                if i < len(decoder_nodes):
                    s.node(decoder_nodes[len(decoder_nodes)-i-1])
        with dot.subgraph() as s:
            s.attr(rank=f'level{depth}')  # Al fondo
            s.node("Bottleneck")

        # --- CONEXIONES ---
        dot.edge("Input", encoder_nodes[0])
        for i in range(len(encoder_nodes) - 1):
            dot.edge(encoder_nodes[i], encoder_nodes[i+1])
        dot.edge(encoder_nodes[-1], "Bottleneck")
        dot.edge("Bottleneck", decoder_nodes[0])
        for i in range(len(decoder_nodes) - 1):
            dot.edge(decoder_nodes[i], decoder_nodes[i+1])
        if final_nodes:
            dot.edge(decoder_nodes[-1], final_nodes[0])
            for i in range(len(final_nodes) - 1):
                dot.edge(final_nodes[i], final_nodes[i+1])
        min_depth = min(len(encoder_nodes), len(decoder_nodes))
        for i in range(min_depth):
            dot.edge(
                encoder_nodes[i],
                decoder_nodes[min_depth-i-1],
                style="dashed", color="#888888", xlabel="skip",
                constraint="false"
            )
        if filename:
            dot.render(filename, view=view, cleanup=True)
        elif view:
            dot.view()
        else:
            for line in dot.source.splitlines():
                print(line)
