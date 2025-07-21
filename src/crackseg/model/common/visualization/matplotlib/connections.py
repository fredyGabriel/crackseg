"""Matplotlib connection drawing functions for U-Net architecture
visualization.

This module contains functions for drawing connections between components
in the U-Net architecture diagram.
"""


def draw_main_path(
    ax, encoder_positions, bottleneck_pos, decoder_positions, final_positions
):
    """Draw main data flow path."""
    # Input to first encoder
    if encoder_positions:
        ax.annotate(
            "",
            xy=encoder_positions[0],
            xytext=(5, 7),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )

        # Encoder chain
        for i in range(len(encoder_positions) - 1):
            ax.annotate(
                "",
                xy=encoder_positions[i + 1],
                xytext=encoder_positions[i],
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
            )

        # Last encoder to bottleneck
        ax.annotate(
            "",
            xy=bottleneck_pos,
            xytext=encoder_positions[-1],
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )
    else:
        # Direct input to bottleneck if no encoder blocks
        ax.annotate(
            "",
            xy=bottleneck_pos,
            xytext=(5, 7),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )

    # Bottleneck to decoder
    if decoder_positions:
        ax.annotate(
            "",
            xy=decoder_positions[0],
            xytext=bottleneck_pos,
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )

        # Decoder chain
        for i in range(len(decoder_positions) - 1):
            ax.annotate(
                "",
                xy=decoder_positions[i + 1],
                xytext=decoder_positions[i],
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
            )

        current_pos = decoder_positions[-1]
    else:
        current_pos = bottleneck_pos

    # Final layers
    if final_positions:
        ax.annotate(
            "",
            xy=final_positions[0],
            xytext=current_pos,
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )

        for i in range(len(final_positions) - 1):
            ax.annotate(
                "",
                xy=final_positions[i + 1],
                xytext=final_positions[i],
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
            )

        current_pos = final_positions[-1]

    # Final to output
    ax.annotate(
        "",
        xy=(5, 1),
        xytext=current_pos,
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
    )


def draw_skip_connections(ax, encoder_positions, decoder_positions):
    """Draw skip connections between encoder and decoder."""
    min_depth = min(len(encoder_positions), len(decoder_positions))

    for i in range(min_depth):
        # Connect encoder[i] to decoder[min_depth-1-i] (reversed order)
        enc_pos = encoder_positions[i]
        dec_pos = decoder_positions[min_depth - 1 - i]

        # Draw curved skip connection
        ax.annotate(
            "",
            xy=dec_pos,
            xytext=enc_pos,
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.5,
                "color": "gray",
                "linestyle": "dashed",
                "alpha": 0.7,
                "connectionstyle": "arc3,rad=0.3",
            },
        )

        # Add "skip" label
        mid_x = (enc_pos[0] + dec_pos[0]) / 2
        mid_y = (enc_pos[1] + dec_pos[1]) / 2 + 0.2
        ax.text(
            mid_x,
            mid_y,
            "skip",
            ha="center",
            va="center",
            fontsize=8,
            color="gray",
            style="italic",
        )
