import logging
import os
from typing import Any

import torch

logger = logging.getLogger(
    __name__
)  # Assuming logger is defined globally or passed


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count the trainable and non-trainable parameters in the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    return trainable, non_trainable


def estimate_receptive_field(encoder) -> dict[str, Any]:
    """Estimate the receptive field size of the model's encoder."""
    depth = getattr(encoder, "depth", None)
    if depth is not None:
        receptive_field_size = 3 + (depth * 4 * 2) - 1
        downsampling_factor = 2**depth
        return {
            "theoretical_rf_size": receptive_field_size,
            "downsampling_factor": downsampling_factor,
            "note": (
                "Theoretical estimate for standard U-Net with 3x3 kernels"
            ),
        }
    else:
        return {
            "note": (
                "Receptive field estimation requires a standard encoder "
                "with known depth"
            )
        }


def estimate_memory_usage(
    model: torch.nn.Module,
    encoder,
    get_output_channels_fn,
    input_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Estimate memory usage for the model."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_bytes + buffer_bytes) / (1024 * 1024)

    if input_shape:
        B, C, H, W = input_shape
        depth = getattr(encoder, "depth", 4)
        encoder_memory = 0
        for i in range(depth):
            features = min(64 * (2**i), 512)
            h, w = H // (2**i), W // (2**i)
            encoder_memory += B * features * h * w * 4
        bottleneck_features = min(64 * (2**depth), 1024)
        bottleneck_h = H // (2**depth)
        bottleneck_w = W // (2**depth)
        bottleneck_memory = (
            B * bottleneck_features * bottleneck_h * bottleneck_w * 4
        )
        decoder_memory = 0
        for i in range(depth):
            j = depth - i - 1
            features = min(64 * (2**j), 512)
            h, w = H // (2**j), W // (2**j)
            decoder_memory += B * features * h * w * 4
        output_memory = B * get_output_channels_fn() * H * W * 4
        activation_mb = (
            encoder_memory + bottleneck_memory + decoder_memory + output_memory
        ) / (1024 * 1024)
        return {
            "model_size_mb": model_size_mb,
            "estimated_activation_mb": activation_mb,
            "total_estimated_mb": model_size_mb + activation_mb,
            "input_shape": input_shape,
        }
    return {
        "model_size_mb": model_size_mb,
        "note": "For activation memory estimates, provide input_shape",
    }


def get_layer_hierarchy(
    encoder, bottleneck, decoder, final_activation=None
) -> list[dict[str, Any]]:
    """Get the hierarchical structure of the model layers."""
    hierarchy = []
    encoder_info = {
        "name": "Encoder",
        "type": encoder.__class__.__name__,
        "params": sum(p.numel() for p in encoder.parameters()),
        "out_channels": encoder.out_channels,
        "skip_channels": encoder.skip_channels,
    }
    if hasattr(encoder, "encoder_blocks"):
        blocks_info = []
        for i, block in enumerate(encoder.encoder_blocks):
            block_info = {
                "name": f"EncoderBlock_{i + 1}",
                "params": sum(p.numel() for p in block.parameters()),
                "in_channels": block.in_channels,
                "out_channels": block.out_channels,
            }
            blocks_info.append(block_info)
        encoder_info["blocks"] = blocks_info
    hierarchy.append(encoder_info)
    bottleneck_info = {
        "name": "Bottleneck",
        "type": bottleneck.__class__.__name__,
        "params": sum(p.numel() for p in bottleneck.parameters()),
        "in_channels": bottleneck.in_channels,
        "out_channels": bottleneck.out_channels,
    }
    hierarchy.append(bottleneck_info)
    decoder_info = {
        "name": "Decoder",
        "type": decoder.__class__.__name__,
        "params": sum(p.numel() for p in decoder.parameters()),
        "in_channels": decoder.in_channels,
        "out_channels": decoder.out_channels,
        "skip_channels": decoder.skip_channels,
    }
    if hasattr(decoder, "decoder_blocks"):
        blocks_info = []
        for i, block in enumerate(decoder.decoder_blocks):
            block_info = {
                "name": f"DecoderBlock_{i + 1}",
                "params": sum(p.numel() for p in block.parameters()),
                "in_channels": block.in_channels,
                "out_channels": block.out_channels,
            }
            blocks_info.append(block_info)
        if hasattr(decoder, "final_conv"):
            final_conv_info = {
                "name": "FinalConv",
                "params": sum(
                    p.numel() for p in decoder.final_conv.parameters()
                ),
                "in_channels": decoder.final_conv.in_channels,
                "out_channels": decoder.final_conv.out_channels,
            }
            blocks_info.append(final_conv_info)
        decoder_info["blocks"] = blocks_info
    hierarchy.append(decoder_info)
    if final_activation is not None:
        activation_info = {
            "name": "FinalActivation",
            "type": final_activation.__class__.__name__,
            "params": sum(p.numel() for p in final_activation.parameters()),
        }
        hierarchy.append(activation_info)
    return hierarchy


# --- Helper functions for diagram rendering ---


def _init_graphviz_digraph() -> (
    Any
):  # Returns Digraph, but avoid type dep if not available
    """Initializes and configures a new Graphviz Digraph object."""
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError(
            "graphviz is required for visualize_architecture. "
            "Install with 'conda install graphviz python-graphviz'."
        ) from exc
    dot = Digraph(comment="U-Net Architecture", format="png", strict=True)
    dot.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.5",
        ranksep="0.5",
        fontsize="12",
    )
    return dot


def _create_io_nodes(dot: Any) -> None:
    """Creates the Input and Output nodes for the U-Net diagram."""
    dot.node(
        "Input",
        "Input\n[Batch, C, H, W]",
        shape="oval",
        style="filled",
        fillcolor="#e0e0e0",
    )
    dot.node(
        "Output",
        "Output\n[Batch, C, H, W]",
        shape="oval",
        style="filled",
        fillcolor="#e0e0e0",
    )


def _extract_component_info(layer_hierarchy: list) -> dict[str, Any]:
    """Extracts encoder_blocks, decoder_blocks, bottleneck, and final_nodes
    info."""
    info: dict[str, Any] = {
        "encoder_blocks": [],
        "decoder_blocks": [],
        "bottleneck_info": None,
        "final_conv_block": None,
        "has_activation": False,
    }
    for layer in layer_hierarchy:
        if layer["name"] == "Encoder" and "blocks" in layer:
            info["encoder_blocks"] = layer["blocks"]
        elif layer["name"] == "Decoder" and "blocks" in layer:
            info["decoder_blocks"] = layer["blocks"]
            for block in layer["blocks"]:
                if block["name"] == "FinalConv":
                    info["final_conv_block"] = block
        elif layer["name"] == "Bottleneck":
            info["bottleneck_info"] = layer
        elif layer["name"] == "FinalActivation":
            info["has_activation"] = True
    return info


def _create_encoder_nodes(
    dot: Any, encoder_blocks: list[dict[str, Any]]
) -> list[str]:
    """Creates and returns names of encoder block nodes."""
    encoder_nodes = []
    with dot.subgraph(name="cluster_encoder") as c:  # type: ignore[union-attr]
        c.attr(rank="same")
        if encoder_blocks:
            for i, block in enumerate(encoder_blocks):
                node_name = f"Enc{i + 1}"
                c.node(
                    node_name,
                    f"EncoderBlock {i + 1}\n"
                    f"({block['in_channels']}→{block['out_channels']})",
                    shape="box",
                    style="filled",
                    fillcolor="#b3cde0",
                )
                encoder_nodes.append(node_name)
        else:
            # Fallback if no detailed blocks, create a single Encoder node
            dot.node(
                "Encoder",
                "Encoder",
                shape="box",
                style="filled",
                fillcolor="#b3cde0",
            )
            encoder_nodes.append("Encoder")
    return encoder_nodes


def _create_bottleneck_node(
    dot: Any, bottleneck_info: dict[str, Any] | None
) -> str:
    """Creates the bottleneck node and returns its name."""
    bn_in = bottleneck_info["in_channels"] if bottleneck_info else "N/A"
    bn_out = bottleneck_info["out_channels"] if bottleneck_info else "N/A"
    dot.node(
        "Bottleneck",
        f"Bottleneck\n({bn_in}→{bn_out})",
        shape="box",
        style="filled",
        fillcolor="#fbb4ae",
    )
    return "Bottleneck"


def _create_decoder_nodes(
    dot: Any, decoder_blocks: list[dict[str, Any]]
) -> list[str]:
    """Creates and returns names of decoder block nodes."""
    decoder_nodes = []
    with dot.subgraph(name="cluster_decoder") as c:  # type: ignore[union-attr]
        c.attr(rank="same")
        if decoder_blocks:
            # Filter out FinalConv if it's part of decoder_blocks for main
            # path rendering
            main_decoder_blocks = [
                b for b in decoder_blocks if b["name"] != "FinalConv"
            ]
            for i, block in enumerate(main_decoder_blocks):
                node_name = f"Dec{i + 1}"
                c.node(
                    node_name,
                    f"DecoderBlock {i + 1}\n"
                    f"({block['in_channels']}→{block['out_channels']})",
                    shape="box",
                    style="filled",
                    fillcolor="#ccebc5",
                )
                decoder_nodes.append(node_name)
        else:
            # Fallback if no detailed blocks
            dot.node(
                "Decoder",
                "Decoder",
                shape="box",
                style="filled",
                fillcolor="#ccebc5",
            )
            decoder_nodes.append("Decoder")
    return decoder_nodes


def _create_final_conv_activation_nodes(
    dot: Any, final_conv_block: dict[str, Any] | None, has_activation: bool
) -> list[str]:
    """Creates FinalConv and Activation nodes if they exist, returns their
    names."""
    final_nodes_sequence = []
    if final_conv_block:
        dot.node(
            "FinalConv",
            f"FinalConv\n({final_conv_block['in_channels']}→"
            f"{final_conv_block['out_channels']})",
            shape="box",
            style="filled",
            fillcolor="#decbe4",
        )
        final_nodes_sequence.append("FinalConv")

    if has_activation:
        dot.node(
            "Activation",
            "Activation",
            shape="ellipse",
            style="filled",
            fillcolor="#fed9a6",
        )
        final_nodes_sequence.append("Activation")
    return final_nodes_sequence


def _connect_main_path(
    dot: Any,
    encoder_node_names: list[str],
    bottleneck_node_name: str,
    decoder_node_names: list[str],
    final_path_node_names: list[str],
) -> None:
    """Connects the main U-Net path: Input -> Encoders -> Bottleneck ->
    Decoders -> FinalConv/Activation -> Output."""
    current_node = "Input"
    if encoder_node_names:
        dot.edge(current_node, encoder_node_names[0])
        for i in range(len(encoder_node_names) - 1):
            dot.edge(encoder_node_names[i], encoder_node_names[i + 1])
        current_node = encoder_node_names[-1]

    dot.edge(current_node, bottleneck_node_name)
    current_node = bottleneck_node_name

    if decoder_node_names:
        dot.edge(current_node, decoder_node_names[0])
        for i in range(len(decoder_node_names) - 1):
            dot.edge(decoder_node_names[i], decoder_node_names[i + 1])
        current_node = decoder_node_names[-1]

    if final_path_node_names:
        dot.edge(current_node, final_path_node_names[0])
        for i in range(len(final_path_node_names) - 1):
            dot.edge(final_path_node_names[i], final_path_node_names[i + 1])
        current_node = final_path_node_names[-1]

    dot.edge(current_node, "Output")


def _connect_skip_connections(
    dot: Any, encoder_node_names: list[str], decoder_node_names: list[str]
) -> None:
    """Connects skip connections between encoder and decoder blocks."""
    min_depth = min(len(encoder_node_names), len(decoder_node_names))
    for i in range(min_depth):
        dot.edge(
            encoder_node_names[i],
            decoder_node_names[
                min_depth - 1 - i
            ],  # Connect to corresponding decoder block (reversed order)
            style="dashed",
            color="#888888",
            xlabel="skip",
            constraint="false",
        )


def _render_or_view_diagram(
    dot: Any, filename: str | None, view: bool
) -> None:
    """Renders the diagram to a file or views it."""
    if filename:
        base_name = os.path.basename(filename)
        name_part = base_name.split(".")[0]
        output_directory = os.path.dirname(filename)
        if not output_directory:
            output_directory = "."
        final_path_base = os.path.join(output_directory, name_part)
        try:
            # Graphviz appends the format automatically
            dot.render(
                filename=final_path_base,
                view=view,
                cleanup=True,
                format=dot.format or "png",
            )
            logger.info(
                "Architecture diagram saved to "
                f"{final_path_base}.{dot.format or 'png'}"
            )
        except Exception as e:
            logger.error(f"Could not render/save architecture diagram: {e}")
    elif view:
        try:
            dot.view(cleanup=True)
            logger.info("Attempting to view architecture diagram.")
        except Exception as e:
            logger.error(f"Could not view architecture diagram: {e}")
    else:
        logger.info(
            "Architecture diagram generated; not saved or viewed "
            "(no filename/view=False)."
        )


# --- Main function ---


def render_unet_architecture_diagram(
    layer_hierarchy: list, filename: str | None = None, view: bool = False
) -> None:
    """Render a U-Net architecture diagram using graphviz from a layer
    hierarchy."""
    dot = _init_graphviz_digraph()
    _create_io_nodes(dot)

    component_info = _extract_component_info(layer_hierarchy)

    encoder_node_names = _create_encoder_nodes(
        dot, component_info["encoder_blocks"]
    )
    bottleneck_node_name = _create_bottleneck_node(
        dot, component_info["bottleneck_info"]
    )
    decoder_node_names = _create_decoder_nodes(
        dot, component_info["decoder_blocks"]
    )
    final_path_node_names = _create_final_conv_activation_nodes(
        dot,
        component_info["final_conv_block"],
        component_info["has_activation"],
    )

    # Note: Graphviz automatic layout handles most positioning.
    # Manual rank positioning can be tricky and might be an advanced
    # refinement if needed.

    _connect_main_path(
        dot,
        encoder_node_names,
        bottleneck_node_name,
        decoder_node_names,
        final_path_node_names,
    )
    if (
        encoder_node_names and decoder_node_names
    ):  # Only connect skips if both encoder and decoder blocks exist
        _connect_skip_connections(dot, encoder_node_names, decoder_node_names)

    _render_or_view_diagram(dot, filename, view)


# Utility to print config with resolved paths
def print_config(config: dict) -> None:
    """Print the configuration with resolved paths."""
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            print_config(value)
        else:
            print(f"{key}: {value}")
