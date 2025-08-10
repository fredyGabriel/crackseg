"""Plotting utilities for prediction visualization helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle


def compute_grid_layout(
    n_items: int, grid_layout: tuple[int, int] | None
) -> tuple[int, int]:
    """Determine (rows, cols) for a grid that fits n_items.

    If grid_layout is provided, validate it fits n_items.
    """
    if grid_layout:
        rows, cols = grid_layout
        if rows * cols < n_items:
            raise ValueError(
                f"Grid layout {grid_layout} too small for {n_items} items"
            )
        return rows, cols

    cols = min(3, n_items)
    rows = (n_items + cols - 1) // cols
    return rows, cols


def reshape_axes_to_2d(axes: Any, rows: int, cols: int) -> np.ndarray:
    """Normalize matplotlib axes to a 2D numpy array shape (rows, cols)."""
    if rows == 1 and cols == 1:
        return np.array([[axes]])
    if rows == 1:
        return axes.reshape(1, -1)
    if cols == 1:
        return axes.reshape(-1, 1)
    return axes


def overlay_mask(
    ax: Axes,
    mask: np.ndarray,
    cmap: str,
    alpha: float,
    base_image: np.ndarray | None = None,
    base_alpha: float = 0.3,
) -> None:
    """Overlay a mask on an axis, optionally with a base image underneath."""
    ax.imshow(mask, cmap=cmap, alpha=alpha)
    if base_image is not None:
        ax.imshow(base_image, alpha=base_alpha)


def build_error_map_and_legend(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> tuple[np.ndarray, Any, list[Rectangle]]:
    """Compute error map (FP/FN/TP) and legend elements.

    Returns:
        - error_map: uint8 map with 0=background, 1=FP (red), 2=FN (blue), 3=TP (green)
        - cmap: ListedColormap for rendering
        - legend_elements: rectangles for legend
    """
    false_positives = pred_mask & ~gt_mask
    false_negatives = ~pred_mask & gt_mask
    true_positives = pred_mask & gt_mask

    error_map = np.zeros_like(gt_mask, dtype=np.uint8)
    error_map[false_positives] = 1
    error_map[false_negatives] = 2
    error_map[true_positives] = 3

    colors = ["black", "red", "blue", "green"]
    cmap = plt.cm.colors.ListedColormap(colors)
    legend_elements = [
        Rectangle(
            (0, 0), 1, 1, facecolor="red", alpha=0.7, label="False Positive"
        ),
        Rectangle(
            (0, 0), 1, 1, facecolor="blue", alpha=0.7, label="False Negative"
        ),
        Rectangle(
            (0, 0), 1, 1, facecolor="green", alpha=0.7, label="True Positive"
        ),
    ]
    return error_map, cmap, legend_elements


def hide_unused_subplots(axes_2d: np.ndarray, used: int) -> None:
    """Hide axes beyond 'used' count in a grid."""
    rows, cols = axes_2d.shape
    for i in range(used, rows * cols):
        r, c = divmod(i, cols)
        axes_2d[r, c].axis("off")
