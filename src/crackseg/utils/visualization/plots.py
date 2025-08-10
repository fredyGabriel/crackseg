"""
Visualization utilities for model predictions and segmentation results.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
# The above suppressions are necessary for matplotlib/numpy due to incomplete
# type stubs

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from .utils.display import (
    BatchDisplayConfig,
    PlottingConfig,
)
from .utils.display import (
    create_overlay_visualization as _create_overlay_visualization,
)
from .utils.display import (
    get_masked_prediction as _get_masked_prediction,
)
from .utils.display import (
    plot_segmentation_comparison as _plot_segmentation_comparison,
)
from .utils.display import (
    visualize_batch_predictions as _visualize_batch_predictions,
)
from .utils.display import (
    visualize_predictions as _visualize_predictions,
)


def visualize_predictions(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    output_dir: str,
    num_samples: int = 5,
    cfg: DictConfig | None = None,
) -> None:
    _visualize_predictions(
        inputs, targets, outputs, output_dir, num_samples, cfg
    )


def plot_segmentation_comparison(
    image: torch.Tensor | np.ndarray[Any, Any],
    prediction: torch.Tensor | np.ndarray[Any, Any],
    target: torch.Tensor | np.ndarray[Any, Any],
    cfg: DictConfig,
    plot_cfg: PlottingConfig | None = None,
):
    _plot_segmentation_comparison(image, prediction, target, cfg, plot_cfg)


def visualize_batch_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    cfg: DictConfig,
    plot_cfg: PlottingConfig | None = None,
    batch_display_cfg: BatchDisplayConfig | None = None,
):
    return _visualize_batch_predictions(
        images, predictions, cfg, plot_cfg, batch_display_cfg
    )


def create_overlay_visualization(
    image_rgb: np.ndarray[Any, Any],
    prediction_mask: np.ndarray[Any, Any],
    cfg: DictConfig,
    alpha: float = 0.4,
    cmap_name: str = "viridis",
):
    return _create_overlay_visualization(
        image_rgb, prediction_mask, cfg, alpha, cmap_name
    )


def get_masked_prediction(pred_array: np.ndarray[Any, Any], cfg: DictConfig):
    return _get_masked_prediction(pred_array, cfg)
