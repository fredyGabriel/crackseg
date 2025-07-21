"""Matplotlib utility functions for U-Net architecture visualization.

This module contains utility functions for data extraction and file handling
in the matplotlib-based U-Net architecture visualization.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_component_info(
    layer_hierarchy: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract component information from layer hierarchy for visualization."""
    component_info = {
        "encoder_blocks": [],
        "bottleneck_info": None,
        "decoder_blocks": [],
        "final_conv_block": None,
        "has_activation": False,
    }

    for layer in layer_hierarchy:
        if layer["name"] == "Encoder" and "blocks" in layer:
            component_info["encoder_blocks"] = layer["blocks"]
        elif layer["name"] == "Bottleneck":
            component_info["bottleneck_info"] = layer
        elif layer["name"] == "Decoder" and "blocks" in layer:
            # Filter out FinalConv from main decoder blocks
            decoder_blocks = []
            for block in layer["blocks"]:
                if block["name"] == "FinalConv":
                    component_info["final_conv_block"] = block
                else:
                    decoder_blocks.append(block)
            component_info["decoder_blocks"] = decoder_blocks
        elif layer["name"] == "FinalActivation":
            component_info["has_activation"] = True

    return component_info


def save_or_show_matplotlib(fig, filename, view):
    """Save figure to file and/or display it."""
    if filename:
        try:
            fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Architecture diagram saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save architecture diagram: {e}")

    if view:
        try:
            import matplotlib.pyplot as plt

            plt.show()
            logger.info("Displaying architecture diagram.")
        except Exception as e:
            logger.error(f"Could not display architecture diagram: {e}")

    if not filename and not view:
        logger.info("Architecture diagram generated; not saved or displayed.")


def get_default_colors():
    """Get default color scheme for U-Net architecture diagram."""
    return {
        "input": "#e0e0e0",
        "encoder": "#b3cde0",
        "bottleneck": "#fbb4ae",
        "decoder": "#ccebc5",
        "final": "#decbe4",
        "activation": "#fed9a6",
        "output": "#e0e0e0",
    }


def get_default_layout():
    """Get default layout parameters for U-Net architecture diagram."""
    return {
        "block_width": 1.5,
        "block_height": 0.8,
        "spacing_y": 1.2,
        "figsize": (12, 8),
    }
