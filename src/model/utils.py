import torch
from typing import Tuple, Dict, Any, List, Optional


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count the trainable and non-trainable parameters in the model."""
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    non_trainable = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    return trainable, non_trainable


def estimate_receptive_field(encoder) -> Dict[str, Any]:
    """Estimate the receptive field size of the model's encoder."""
    depth = getattr(encoder, 'depth', None)
    if depth is not None:
        receptive_field_size = 3 + (depth * 4 * 2) - 1
        downsampling_factor = 2 ** depth
        return {
            "theoretical_rf_size": receptive_field_size,
            "downsampling_factor": downsampling_factor,
            "note": (
                "Theoretical estimate for standard U-Net with 3x3 kernels"
            )
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
    input_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """Estimate memory usage for the model."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_bytes + buffer_bytes) / (1024 * 1024)

    if input_shape:
        B, C, H, W = input_shape
        depth = getattr(encoder, 'depth', 4)
        encoder_memory = 0
        for i in range(depth):
            features = min(64 * (2 ** i), 512)
            h, w = H // (2 ** i), W // (2 ** i)
            encoder_memory += B * features * h * w * 4
        bottleneck_features = min(64 * (2 ** depth), 1024)
        bottleneck_h = H // (2 ** depth)
        bottleneck_w = W // (2 ** depth)
        bottleneck_memory = (
            B * bottleneck_features * bottleneck_h * bottleneck_w * 4
        )
        decoder_memory = 0
        for i in range(depth):
            j = depth - i - 1
            features = min(64 * (2 ** j), 512)
            h, w = H // (2 ** j), W // (2 ** j)
            decoder_memory += B * features * h * w * 4
        output_memory = B * get_output_channels_fn() * H * W * 4
        activation_mb = (
            encoder_memory + bottleneck_memory + decoder_memory + output_memory
        ) / (1024 * 1024)
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


def get_layer_hierarchy(
    encoder,
    bottleneck,
    decoder,
    final_activation=None
) -> List[Dict[str, Any]]:
    """Get the hierarchical structure of the model layers."""
    hierarchy = []
    encoder_info = {
        "name": "Encoder",
        "type": encoder.__class__.__name__,
        "params": sum(p.numel() for p in encoder.parameters()),
        "out_channels": encoder.out_channels,
        "skip_channels": encoder.skip_channels
    }
    if hasattr(encoder, 'encoder_blocks'):
        blocks_info = []
        for i, block in enumerate(encoder.encoder_blocks):
            block_info = {
                "name": f"EncoderBlock_{i+1}",
                "params": sum(p.numel() for p in block.parameters()),
                "in_channels": block.in_channels,
                "out_channels": block.out_channels
            }
            blocks_info.append(block_info)
        encoder_info["blocks"] = blocks_info
    hierarchy.append(encoder_info)
    bottleneck_info = {
        "name": "Bottleneck",
        "type": bottleneck.__class__.__name__,
        "params": sum(p.numel() for p in bottleneck.parameters()),
        "in_channels": bottleneck.in_channels,
        "out_channels": bottleneck.out_channels
    }
    hierarchy.append(bottleneck_info)
    decoder_info = {
        "name": "Decoder",
        "type": decoder.__class__.__name__,
        "params": sum(p.numel() for p in decoder.parameters()),
        "in_channels": decoder.in_channels,
        "out_channels": decoder.out_channels,
        "skip_channels": decoder.skip_channels
    }
    if hasattr(decoder, 'decoder_blocks'):
        blocks_info = []
        for i, block in enumerate(decoder.decoder_blocks):
            block_info = {
                "name": f"DecoderBlock_{i+1}",
                "params": sum(p.numel() for p in block.parameters()),
                "in_channels": block.in_channels,
                "out_channels": block.out_channels
            }
            blocks_info.append(block_info)
        if hasattr(decoder, 'final_conv'):
            final_conv_info = {
                "name": "FinalConv",
                "params": sum(
                    p.numel() for p in decoder.final_conv.parameters()
                ),
                "in_channels": decoder.final_conv.in_channels,
                "out_channels": decoder.final_conv.out_channels
            }
            blocks_info.append(final_conv_info)
        decoder_info["blocks"] = blocks_info
    hierarchy.append(decoder_info)
    if final_activation is not None:
        activation_info = {
            "name": "FinalActivation",
            "type": final_activation.__class__.__name__,
            "params": sum(p.numel() for p in final_activation.parameters())
        }
        hierarchy.append(activation_info)
    return hierarchy


def render_unet_architecture_diagram(
    layer_hierarchy: list,
    filename: str = None,
    view: bool = False
) -> None:
    """Render a U-Net architecture diagram using graphviz from a layer
    hierarchy."""
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "graphviz is required for visualize_architecture. "
            "Install with 'conda install graphviz python-graphviz'."
        )
    dot = Digraph(comment="U-Net Architecture", format="png", strict=True)
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.5",
             ranksep="0.5", fontsize="12")
    # Nodo de entrada
    dot.node(
        "Input",
        "Input\n[Batch, C, H, W]",
        shape="oval",
        style="filled",
        fillcolor="#e0e0e0"
    )
    encoder_blocks = []
    decoder_blocks = []
    bottleneck = None
    final_nodes = []
    # Buscar bloques en la jerarquía
    for layer in layer_hierarchy:
        if layer["name"] == "Encoder" and "blocks" in layer:
            encoder_blocks = layer["blocks"]
        elif layer["name"] == "Decoder" and "blocks" in layer:
            decoder_blocks = layer["blocks"]
        elif layer["name"] == "Bottleneck":
            bottleneck = layer
        elif layer["name"] == "FinalActivation":
            final_nodes.append("Activation")
    # Encoder
    encoder_nodes = []
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(rank='same')
        if encoder_blocks:
            for i, block in enumerate(encoder_blocks):
                node_name = f"Enc{i+1}"
                c.node(
                    node_name,
                    f"EncoderBlock {i+1}\n({block['in_channels']}→"
                    f"{block['out_channels']})",
                    shape="box",
                    style="filled",
                    fillcolor="#b3cde0"
                )
                encoder_nodes.append(node_name)
        else:
            c.node("Encoder", "Encoder", shape="box", style="filled",
                   fillcolor="#b3cde0")
            encoder_nodes.append("Encoder")
    # Bottleneck
    dot.node(
        "Bottleneck",
        f"Bottleneck\n({bottleneck['in_channels']}→"
        f"{bottleneck['out_channels']})",
        shape="box",
        style="filled",
        fillcolor="#fbb4ae"
    )
    # Decoder
    decoder_nodes = []
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(rank='same')
        if decoder_blocks:
            for i, block in enumerate(decoder_blocks):
                node_name = f"Dec{i+1}"
                c.node(
                    node_name,
                    f"DecoderBlock {i+1}\n({block['in_channels']}→"
                    f"{block['out_channels']})",
                    shape="box",
                    style="filled",
                    fillcolor="#ccebc5"
                )
                decoder_nodes.append(node_name)
        else:
            c.node("Decoder", "Decoder", shape="box", style="filled",
                   fillcolor="#ccebc5")
            decoder_nodes.append("Decoder")
    # Final conv y activación
    for layer in layer_hierarchy:
        if layer["name"] == "Decoder" and "blocks" in layer:
            for block in layer["blocks"]:
                if block["name"] == "FinalConv":
                    dot.node(
                        "FinalConv",
                        f"FinalConv\n({block['in_channels']}→"
                        f"{block['out_channels']})",
                        shape="box",
                        style="filled",
                        fillcolor="#decbe4"
                    )
                    final_nodes.append("FinalConv")
    if final_nodes and final_nodes[-1] != "Activation":
        # Si hay activación final
        dot.node(
            "Activation",
            "Activation",
            shape="ellipse",
            style="filled",
            fillcolor="#fed9a6"
        )
    dot.node(
        "Output",
        "Output\n[Batch, C, H, W]",
        shape="oval",
        style="filled",
        fillcolor="#e0e0e0"
    )
    # Posicionamiento
    depth = max(len(encoder_nodes), len(decoder_nodes))
    for i in range(depth):
        with dot.subgraph() as s:
            s.attr(rank=f'level{i}')
            if i < len(encoder_nodes):
                s.node(encoder_nodes[i])
            if i < len(decoder_nodes):
                s.node(decoder_nodes[len(decoder_nodes)-i-1])
    with dot.subgraph() as s:
        s.attr(rank=f'level{depth}')
        s.node("Bottleneck")
    # Conexiones
    dot.edge("Input", encoder_nodes[0])
    for i in range(len(encoder_nodes) - 1):
        dot.edge(
            encoder_nodes[i],
            encoder_nodes[i+1]
        )
    dot.edge(encoder_nodes[-1], "Bottleneck")
    dot.edge("Bottleneck", decoder_nodes[0])
    for i in range(len(decoder_nodes) - 1):
        dot.edge(
            decoder_nodes[i],
            decoder_nodes[i+1]
        )
    if final_nodes:
        dot.edge(decoder_nodes[-1], final_nodes[0])
        for i in range(len(final_nodes) - 1):
            dot.edge(final_nodes[i], final_nodes[i+1])
    min_depth = min(len(encoder_nodes), len(decoder_nodes))
    for i in range(min_depth):
        dot.edge(
            encoder_nodes[i],
            decoder_nodes[min_depth-i-1],
            style="dashed",
            color="#888888",
            xlabel="skip",
            constraint="false"
        )
    last_node = final_nodes[-1] if final_nodes else decoder_nodes[-1]
    dot.edge(
        last_node,
        "Output"
    )
    if filename:
        dot.render(filename, view=view, cleanup=True)
    elif view:
        dot.view()
    else:
        for line in dot.source.splitlines():
            print(line)
