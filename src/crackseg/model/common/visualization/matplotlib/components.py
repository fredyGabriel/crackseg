"""Matplotlib drawing components for U-Net architecture visualization.

This module contains the individual drawing functions for different
components of the U-Net architecture diagram.
"""

import matplotlib.patches as patches


def draw_io_nodes(ax, colors, block_width, block_height):
    """Draw input and output nodes."""
    # Input node (top center)
    input_rect = patches.FancyBboxPatch(
        (4.25, 7),
        block_width,
        block_height,
        boxstyle="round,pad=0.1",
        facecolor=colors["input"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(input_rect)
    ax.text(
        5,
        7.4,
        "Input\n[B,C,H,W]",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Output node (bottom center)
    output_rect = patches.FancyBboxPatch(
        (4.25, 0.2),
        block_width,
        block_height,
        boxstyle="round,pad=0.1",
        facecolor=colors["output"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(output_rect)
    ax.text(
        5,
        0.6,
        "Output\n[B,C,H,W]",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def draw_encoder_blocks(
    ax, encoder_blocks, colors, block_width, block_height, spacing_y
):
    """Draw encoder blocks on the left side."""
    positions = []
    start_y = 6
    x = 1

    for i, block in enumerate(encoder_blocks):
        y = start_y - (i * spacing_y)

        # Draw block
        rect = patches.FancyBboxPatch(
            (x, y),
            block_width,
            block_height,
            boxstyle="round,pad=0.05",
            facecolor=colors["encoder"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        # Add text
        text = f"Enc{i + 1}\n({block['in_channels']}→{block['out_channels']})"
        ax.text(
            x + block_width / 2,
            y + block_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

        positions.append((x + block_width / 2, y + block_height / 2))

    return positions


def draw_bottleneck(ax, bottleneck_info, colors, block_width, block_height):
    """Draw bottleneck block at center bottom."""
    x, y = 4.25, 2.5

    if bottleneck_info:
        in_ch = bottleneck_info.get("in_channels", "N/A")
        out_ch = bottleneck_info.get("out_channels", "N/A")
        text = f"Bottleneck\n({in_ch}→{out_ch})"
    else:
        text = "Bottleneck"

    # Draw block
    rect = patches.FancyBboxPatch(
        (x, y),
        block_width,
        block_height,
        boxstyle="round,pad=0.05",
        facecolor=colors["bottleneck"],
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect)

    # Add text
    ax.text(
        x + block_width / 2,
        y + block_height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    return (x + block_width / 2, y + block_height / 2)


def draw_decoder_blocks(
    ax, decoder_blocks, colors, block_width, block_height, spacing_y
):
    """Draw decoder blocks on the right side."""
    positions = []
    start_y = 6
    x = 7.5

    for i, block in enumerate(decoder_blocks):
        y = start_y - (i * spacing_y)

        # Draw block
        rect = patches.FancyBboxPatch(
            (x, y),
            block_width,
            block_height,
            boxstyle="round,pad=0.05",
            facecolor=colors["decoder"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        # Add text
        text = f"Dec{i + 1}\n({block['in_channels']}→{block['out_channels']})"
        ax.text(
            x + block_width / 2,
            y + block_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

        positions.append((x + block_width / 2, y + block_height / 2))

    return positions


def draw_final_layers(
    ax, final_conv_block, has_activation, colors, block_width, block_height
):
    """Draw final convolution and activation layers."""
    positions = []
    x = 4.25
    y = 1.4

    if final_conv_block:
        # Draw final conv
        rect = patches.FancyBboxPatch(
            (x, y),
            block_width,
            block_height,
            boxstyle="round,pad=0.05",
            facecolor=colors["final"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        in_ch = final_conv_block.get("in_channels", "N/A")
        out_ch = final_conv_block.get("out_channels", "N/A")
        text = f"FinalConv\n({in_ch}→{out_ch})"
        ax.text(
            x + block_width / 2,
            y + block_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

        positions.append((x + block_width / 2, y + block_height / 2))
        y -= 0.9

    if has_activation:
        # Draw activation
        rect = patches.FancyBboxPatch(
            (x, y),
            block_width,
            block_height,
            boxstyle="round,pad=0.05",
            facecolor=colors["activation"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        ax.text(
            x + block_width / 2,
            y + block_height / 2,
            "Activation",
            ha="center",
            va="center",
            fontsize=9,
        )

        positions.append((x + block_width / 2, y + block_height / 2))

    return positions
