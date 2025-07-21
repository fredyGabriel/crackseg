"""Graphviz-based U-Net architecture visualization.

This module provides graphviz-based rendering for U-Net architecture diagrams.
It's maintained as a legacy fallback option (ADR-001) for compatibility.
"""

import logging
import os
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)


class DigraphProtocol(Protocol):
    """Protocol for Graphviz Digraph-like objects."""

    format: str | None

    def node(self, name: str, label: str, **attrs: Any) -> None: ...
    def edge(self, source: str, target: str, **attrs: Any) -> None: ...
    def attr(self, **attrs: Any) -> None: ...
    def subgraph(self, name: str) -> Any: ...
    def render(self, filename: str, **kwargs: Any) -> str: ...
    def view(self, **kwargs: Any) -> None: ...


def render_unet_architecture_graphviz(
    layer_hierarchy: list[dict[str, Any]],
    filename: str | None = None,
    view: bool = False,
) -> None:
    """Render a U-Net architecture diagram using graphviz from a layer
    hierarchy.

    Legacy graphviz implementation maintained for compatibility.
    Consider using render_unet_architecture_matplotlib() instead (ADR-001).
    """
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


def _init_graphviz_digraph() -> DigraphProtocol:
    """Initializes and configures a new Graphviz Digraph object."""
    try:
        # Dynamic import to avoid type checking issues
        import graphviz  # type: ignore[import-untyped]

        digraph_class = graphviz.Digraph  # type: ignore[reportUnknownMemberType]
    except ImportError as exc:
        raise ImportError(
            "graphviz is required for visualize_architecture. "
            "Install with 'conda install graphviz python-graphviz'."
        ) from exc

    # Use cast to ensure type safety with protocol
    dot = cast(
        DigraphProtocol,
        digraph_class(comment="U-Net Architecture", format="png", strict=True),
    )
    dot.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.5",
        ranksep="0.5",
        fontsize="12",
    )
    return dot


def _create_io_nodes(dot: DigraphProtocol) -> None:
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


def _extract_component_info(
    layer_hierarchy: list[dict[str, Any]],
) -> dict[str, Any]:
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
    dot: DigraphProtocol, encoder_blocks: list[dict[str, Any]]
) -> list[str]:
    """Creates and returns names of encoder block nodes."""
    encoder_nodes: list[str] = []
    with dot.subgraph(name="cluster_encoder") as c:
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
    dot: DigraphProtocol, bottleneck_info: dict[str, Any] | None
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
    dot: DigraphProtocol, decoder_blocks: list[dict[str, Any]]
) -> list[str]:
    """Creates and returns names of decoder block nodes."""
    decoder_nodes: list[str] = []
    with dot.subgraph(name="cluster_decoder") as c:
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
    dot: DigraphProtocol,
    final_conv_block: dict[str, Any] | None,
    has_activation: bool,
) -> list[str]:
    """Creates FinalConv and Activation nodes if they exist, returns their
    names."""
    final_nodes_sequence: list[str] = []
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
    dot: DigraphProtocol,
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
    dot: DigraphProtocol,
    encoder_node_names: list[str],
    decoder_node_names: list[str],
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
    dot: DigraphProtocol, filename: str | None, view: bool
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
