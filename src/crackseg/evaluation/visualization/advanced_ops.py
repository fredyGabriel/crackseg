"""Heavy plotting operations for AdvancedPredictionVisualizer.

This module contains standalone functions used by the class methods to keep
the class definition compact and under line limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from plotly.graph_objs import Figure as PlotlyFigure

from .templates.prediction_template import PredictionVisualizationTemplate
from .utils.images import load_image_rgb, save_figure
from .utils.plot_utils import (
    build_error_map_and_legend,
    compute_grid_layout,
    hide_unused_subplots,
    overlay_mask,
    reshape_axes_to_2d,
)


def create_comparison_grid(
    template: PredictionVisualizationTemplate,
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
    max_images: int = 9,
    show_metrics: bool = True,
    show_confidence: bool = True,
    grid_layout: tuple[int, int] | None = None,
) -> Figure | PlotlyFigure:
    if not results:
        raise ValueError("No results provided for comparison")

    results = results[:max_images]
    n_results = len(results)
    rows, cols = compute_grid_layout(n_results, grid_layout)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = reshape_axes_to_2d(axes, rows, cols)

    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        original_image = load_image_rgb(result["image_path"])
        ax.imshow(original_image)
        if "prediction_mask" in result:
            overlay_mask(ax, result["prediction_mask"], cmap="Reds", alpha=0.7)
        if "ground_truth_mask" in result:
            overlay_mask(
                ax, result["ground_truth_mask"], cmap="Blues", alpha=0.5
            )
        title = Path(result["image_path"]).stem
        if show_metrics and "iou" in result:
            title += f"\nIoU: {result['iou']:.3f}"
        if show_metrics and "dice" in result:
            title += f" Dice: {result['dice']:.3f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    hide_unused_subplots(axes, n_results)
    fig = template.apply_template(fig)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def create_confidence_map(
    template: PredictionVisualizationTemplate,
    result: dict[str, Any],
    save_path: str | Path | None = None,
    show_original: bool = True,
    show_contours: bool = True,
) -> Figure | PlotlyFigure:
    if "probability_mask" not in result:
        raise ValueError(
            "Result must contain 'probability_mask' for confidence map"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    original_image = load_image_rgb(result["image_path"])
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    prob_mask = result["probability_mask"]
    im = axes[1].imshow(prob_mask, cmap="viridis", alpha=0.8)
    if show_original:
        axes[1].imshow(original_image, alpha=0.3)
    axes[1].set_title("Confidence Map", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if show_contours:
        contours = axes[1].contour(
            prob_mask, levels=10, colors="white", alpha=0.5, linewidths=0.5
        )
        axes[1].clabel(contours, inline=True, fontsize=8)

    fig = template.apply_template(fig)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def create_error_analysis(
    template: PredictionVisualizationTemplate,
    result: dict[str, Any],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    if "ground_truth_mask" not in result or "prediction_mask" not in result:
        raise ValueError(
            "Result must contain both 'ground_truth_mask' and 'prediction_mask'"
        )

    gt_mask = result["ground_truth_mask"]
    pred_mask = result["prediction_mask"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    original_image = load_image_rgb(result["image_path"])
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontweight="bold")
    axes[0, 0].axis("off")
    overlay_mask(
        axes[0, 1], gt_mask, cmap="Blues", alpha=0.7, base_image=original_image
    )
    axes[0, 1].set_title("Ground Truth", fontweight="bold")
    axes[0, 1].axis("off")
    overlay_mask(
        axes[1, 0],
        pred_mask,
        cmap="Reds",
        alpha=0.7,
        base_image=original_image,
    )
    axes[1, 0].set_title("Prediction", fontweight="bold")
    axes[1, 0].axis("off")
    error_map, cmap, legend_elements = build_error_map_and_legend(
        gt_mask, pred_mask
    )
    overlay_mask(
        axes[1, 1], error_map, cmap=cmap, alpha=0.7, base_image=original_image
    )
    axes[1, 1].set_title("Error Analysis", fontweight="bold")
    axes[1, 1].axis("off")
    axes[1, 1].legend(handles=legend_elements, loc="upper right")
    fig = template.apply_template(fig)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def create_segmentation_overlay(
    template: PredictionVisualizationTemplate,
    result: dict[str, Any],
    save_path: str | Path | None = None,
    show_confidence: bool = True,
) -> Figure | PlotlyFigure:
    original_image = load_image_rgb(result["image_path"])
    has_pred = "prediction_mask" in result
    has_gt = "ground_truth_mask" in result
    has_confidence = show_confidence and "probability_mask" in result
    num_plots = (
        1
        + (1 if has_pred else 0)
        + (1 if has_gt else 0)
        + (1 if has_confidence else 0)
    )
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    plot_idx = 0
    axes[plot_idx].imshow(original_image)
    axes[plot_idx].set_title("Original Image", fontweight="bold")
    axes[plot_idx].axis("off")
    plot_idx += 1
    if has_pred:
        overlay_mask(
            axes[plot_idx],
            result["prediction_mask"],
            cmap="Reds",
            alpha=0.7,
            base_image=original_image,
        )
        axes[plot_idx].set_title("Prediction Overlay", fontweight="bold")
        axes[plot_idx].axis("off")
        plot_idx += 1
    if has_gt:
        overlay_mask(
            axes[plot_idx],
            result["ground_truth_mask"],
            cmap="Blues",
            alpha=0.7,
            base_image=original_image,
        )
        axes[plot_idx].set_title("Ground Truth Overlay", fontweight="bold")
        axes[plot_idx].axis("off")
        plot_idx += 1
    if has_confidence:
        im = axes[plot_idx].imshow(
            result["probability_mask"], cmap="viridis", alpha=0.8
        )
        axes[plot_idx].imshow(original_image, alpha=0.2)
        axes[plot_idx].set_title("Confidence Overlay", fontweight="bold")
        axes[plot_idx].axis("off")
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
    fig = template.apply_template(fig)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def create_tabular_comparison(
    template: PredictionVisualizationTemplate,
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> Figure | PlotlyFigure:
    if not results:
        raise ValueError("No results provided for tabular comparison")
    metrics_data = []
    for result in results:
        row = {"Image": Path(result["image_path"]).stem}
        for metric in ["iou", "dice", "precision", "recall", "f1"]:
            if metric in result:
                row[metric.upper()] = f"{result[metric]:.3f}"
        metrics_data.append(row)
    fig, ax = plt.subplots(figsize=(12, len(metrics_data) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")
    table_data = []
    headers = list(metrics_data[0].keys())
    for row in metrics_data:
        table_data.append([row[header] for header in headers])
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=Bbox.from_bounds(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
    ax.set_title(
        "Prediction Metrics Comparison", fontweight="bold", fontsize=14, pad=20
    )
    fig = template.apply_template(fig)
    if save_path:
        save_figure(fig, save_path)
    return fig
