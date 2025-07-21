"""Matplotlib-based U-Net architecture visualization.

This module provides matplotlib-based rendering for U-Net architecture
diagrams. It's the preferred visualization method (ADR-001) due to better
cross-platform compatibility and simpler dependencies.
"""

import logging
from typing import Any

from .matplotlib.components import (
    draw_bottleneck,
    draw_decoder_blocks,
    draw_encoder_blocks,
    draw_final_layers,
    draw_io_nodes,
)
from .matplotlib.connections import draw_main_path, draw_skip_connections
from .matplotlib.utils import (
    extract_component_info,
    get_default_colors,
    get_default_layout,
    save_or_show_matplotlib,
)

logger = logging.getLogger(__name__)


def render_unet_architecture_matplotlib(
    layer_hierarchy: list[dict[str, Any]],
    filename: str | None = None,
    view: bool = False,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Render U-Net architecture diagram using matplotlib.

    Creates a U-shaped block diagram showing the model architecture with
    component relationships, skip connections, and data flow using matplotlib
    instead of graphviz. This provides the same functionality with simpler
    dependencies.

    Args:
        layer_hierarchy: Model layer information from get_layer_hierarchy()
        filename: Optional output filename for saving diagram.
            If None, generates temporary file or shows plot.
            Supports formats: .png, .pdf, .svg, .jpg
        view: If True, automatically displays the plot.
            Uses matplotlib.pyplot.show() for display.
        figsize: Figure size as (width, height) tuple in inches.
            Default: (12, 8) provides good aspect ratio for U-Net layout.

    Examples:
        >>> # Generate and save diagram
        >>> hierarchy = get_layer_hierarchy(encoder, bottleneck, decoder)
        >>> render_unet_architecture_matplotlib(
        ...     hierarchy, "model_arch.png", view=True
        ... )

        >>> # Quick preview without saving
        >>> render_unet_architecture_matplotlib(hierarchy, view=True)

        >>> # Custom size for high-resolution output
        >>> render_unet_architecture_matplotlib(
        ...     hierarchy, "paper_figure.pdf", figsize=(16, 10)
        ... )

    Features:
        - U-shaped layout showing encoder-decoder symmetry
        - Component blocks with type and channel information
        - Skip connection arrows showing data flow
        - Color coding for different component types
        - Parameter count annotations
        - High-quality output for publications

    Requirements:
        - matplotlib package (already in environment)
        - No additional system dependencies required

    Notes:
        - Replaces graphviz dependency (ADR-001)
        - Maintains visual compatibility with existing workflow
        - Supports all matplotlib output formats
        - Better cross-platform compatibility than graphviz
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for architecture visualization. "
            "It should be available in the crackseg environment."
        ) from exc

    # Extract component information
    component_info = extract_component_info(layer_hierarchy)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Get default colors and layout
    colors = get_default_colors()
    layout = get_default_layout()
    layout["figsize"] = figsize

    # Draw input/output nodes
    draw_io_nodes(ax, colors, layout["block_width"], layout["block_height"])

    # Draw encoder blocks (left side, top to bottom)
    encoder_positions = draw_encoder_blocks(
        ax,
        component_info["encoder_blocks"],
        colors,
        layout["block_width"],
        layout["block_height"],
        layout["spacing_y"],
    )

    # Draw bottleneck (center bottom)
    bottleneck_pos = draw_bottleneck(
        ax,
        component_info["bottleneck_info"],
        colors,
        layout["block_width"],
        layout["block_height"],
    )

    # Draw decoder blocks (right side, bottom to top)
    decoder_positions = draw_decoder_blocks(
        ax,
        component_info["decoder_blocks"],
        colors,
        layout["block_width"],
        layout["block_height"],
        layout["spacing_y"],
    )

    # Draw final layers if they exist
    final_positions = draw_final_layers(
        ax,
        component_info["final_conv_block"],
        component_info["has_activation"],
        colors,
        layout["block_width"],
        layout["block_height"],
    )

    # Draw connections
    draw_main_path(
        ax,
        encoder_positions,
        bottleneck_pos,
        decoder_positions,
        final_positions,
    )

    # Draw skip connections
    if encoder_positions and decoder_positions:
        draw_skip_connections(ax, encoder_positions, decoder_positions)

    # Add title
    plt.title("U-Net Architecture", fontsize=16, fontweight="bold", pad=20)

    # Save or show the plot
    save_or_show_matplotlib(fig, filename, view)
