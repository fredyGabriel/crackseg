import logging
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import DictConfig
from PIL import Image


@dataclass
class PlottingConfig:
    """Configuration for plotting visualizations."""

    title_prefix: str = ""
    fig_size: tuple[int, int] = (12, 4)
    save_path: str | None = None
    show_plot: bool = True


@dataclass
class BatchDisplayConfig:
    """Configuration for displaying batch predictions."""

    targets: torch.Tensor | None = None
    max_samples: int = 5
    fig_size_per_sample: tuple[int, int] = (10, 3)


@dataclass
class GridDisplayConfig:
    """Configuration for displaying results in a grid."""

    targets: list[np.ndarray] | None = None
    num_cols: int = 3
    fig_size_per_row: tuple[int, int] = (15, 5)
    titles: list[str] | None = None


def visualize_predictions(  # noqa: PLR0913
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    output_dir: str,
    num_samples: int = 5,
    cfg: DictConfig | None = None,
) -> None:
    """
    Create and save visualizations of model predictions.

    Args:
        inputs: Input images
        targets: Ground truth masks
        outputs: Model predictions
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        cfg: Optional Hydra config for threshold
    """
    # Limit number of samples
    num_samples = min(num_samples, len(inputs))

    # Apply threshold to outputs (assuming sigmoid activation)
    threshold = cfg.thresholds.metric if cfg is not None else 0.5
    binary_outputs = (outputs > threshold).float()

    # Create directory if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Configure plotting
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get current sample
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()
        binary_pred = binary_outputs[i].squeeze().numpy()

        # Denormalize image if it's normalized to [-1, 1]
        if img.min() < 0:
            img = (img + 1) / 2
        # Clip to [0, 1] range
        img = np.clip(img, 0, 1)

        # Plot original image, ground truth, prediction, and binary prediction
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(img)
        plt.title(f"Image {i + 1}")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction (Raw)")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(binary_pred, cmap="gray")
        plt.title("Prediction (Binary)")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "prediction_samples.png"), dpi=200)
    plt.close()

    # Save individual samples for detailed inspection
    for i in range(num_samples):
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()

        # Denormalize image
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # Create overlay visualization (prediction contour on image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Create contour of the prediction
        threshold = cfg.thresholds.metric if cfg is not None else 0.5
        masked_pred = np.ma.masked_where(pred < threshold, pred)
        ax.imshow(masked_pred, cmap="jet", alpha=0.4)

        plt.title(f"Sample {i + 1}: Image with crack overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, f"overlay_sample_{i + 1}.png"), dpi=200
        )
        plt.close()


def plot_segmentation_comparison(
    image: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    cfg: DictConfig,
    plot_cfg: PlottingConfig | None = None,
):
    if plot_cfg is None:
        plot_cfg = PlottingConfig()
    threshold = cfg.thresholds.metric
    num_channel_dim = (
        cfg.visualization.num_cols
    )  # Use num_cols as a descriptive constant for channel dim if appropriate
    if isinstance(prediction, torch.Tensor):
        binary_prediction = (prediction > threshold).float().cpu().numpy()
        if prediction.shape[0] == num_channel_dim - (
            cfg.visualization.num_cols_no_targets - 1
        ):  # 3-2+1=2, so if shape[0]==1
            binary_prediction = binary_prediction.squeeze(0)
    elif isinstance(prediction, np.ndarray):
        binary_prediction = (prediction > threshold).astype(float)
        if prediction.ndim == num_channel_dim and prediction.shape[
            0
        ] == num_channel_dim - (
            cfg.visualization.num_cols_no_targets - 1
        ):  # Check for channel dim
            binary_prediction = binary_prediction.squeeze(0)
    else:
        raise TypeError("Prediction must be a PyTorch Tensor or NumPy array.")

    fig, axs = plt.subplots(1, 2, figsize=plot_cfg.fig_size)
    axs[0].imshow(image)
    axs[0].set_title(f"{plot_cfg.title_prefix}Original Image")
    axs[0].axis("off")

    axs[1].imshow(binary_prediction, cmap="gray")
    axs[1].set_title(f"{plot_cfg.title_prefix}Prediction (Binary)")
    axs[1].axis("off")

    plt.tight_layout()
    if plot_cfg.save_path:
        plt.savefig(plot_cfg.save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved visualization to {plot_cfg.save_path}")
    if plot_cfg.show_plot:
        plt.show()
    plt.close(fig)


def visualize_batch_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    cfg: DictConfig,
    plot_cfg: PlottingConfig | None = None,
    batch_display_cfg: BatchDisplayConfig | None = None,
):
    if plot_cfg is None:
        plot_cfg = PlottingConfig()
    if batch_display_cfg is None:
        batch_display_cfg = BatchDisplayConfig()
    threshold = cfg.thresholds.metric
    binary_predictions = (predictions > threshold).float()

    targets = batch_display_cfg.targets
    max_samples = batch_display_cfg.max_samples
    fig_size_per_sample = batch_display_cfg.fig_size_per_sample

    num_cols = (
        cfg.visualization.num_cols
        if targets is not None
        else cfg.visualization.num_cols_no_targets
    )
    num_rows = math.ceil(max_samples / num_cols) if num_cols > 0 else 0

    fig_width = fig_size_per_sample[0]
    fig_height = (
        (num_rows * fig_size_per_sample[1])
        if num_cols > 0
        else fig_size_per_sample[1]
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_rows, num_cols),
        axes_pad=0.3,
        share_all=True,
    )

    for i in range(min(max_samples, images.size(0))):
        row = i // num_cols if num_cols > 0 else 0
        col_offset = (i % num_cols) * num_cols if num_cols > 0 else 0

        ax_img_idx = row * num_cols + col_offset
        ax_pred_idx = row * num_cols + col_offset + 1

        img = images[i].permute(1, 2, 0).cpu().numpy()
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = np.clip(img, 0, 1)

        grid[ax_img_idx].imshow(img)
        grid[ax_img_idx].set_title(f"Image {i + 1}")
        grid[ax_img_idx].axis("off")

        ax_pred = grid[ax_pred_idx]
        ax_pred.imshow(
            binary_predictions[i].squeeze().cpu().numpy(), cmap="gray"
        )
        ax_pred.set_title(f"Prediction {i + 1}")
        ax_pred.axis("off")

        if targets is not None and i < targets.size(0):
            ax_target_idx = row * num_cols + col_offset + 2
            target_img = targets[i].squeeze().cpu().numpy()
            grid[ax_target_idx].imshow(target_img, cmap="gray")
            grid[ax_target_idx].set_title(f"Target {i + 1}")
            grid[ax_target_idx].axis("off")

    start_idx_unused_axes = min(max_samples, images.size(0)) * num_cols
    for j in range(start_idx_unused_axes, len(grid)):
        grid[j].axis("off")

    plt.tight_layout()
    if plot_cfg.save_path:
        plt.savefig(plot_cfg.save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved visualization to {plot_cfg.save_path}")

    if plot_cfg.show_plot:
        plt.show()

    plt.close(fig)
    return fig


def create_overlay_visualization(
    image_rgb: np.ndarray,
    prediction_mask: np.ndarray,
    cfg: DictConfig,
    alpha: float = 0.4,
    cmap_name: str = "viridis",
):
    """Creates an overlay of a prediction mask on an RGB image."""
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != cfg.visualization.num_cols
        or image_rgb.shape[2] != cfg.visualization.num_cols
    ):
        raise ValueError("image_rgb must be a HxWx3 NumPy array.")
    if (
        not isinstance(prediction_mask, np.ndarray)
        or prediction_mask.ndim != cfg.visualization.num_cols_no_targets
    ):
        raise ValueError("prediction_mask must be a HxW NumPy array.")
    if image_rgb.shape[:2] != prediction_mask.shape:
        # Try to resize prediction_mask if shapes don't match
        prediction_mask = np.array(
            Image.fromarray(prediction_mask).resize(
                image_rgb.shape[:2][::-1], Image.NEAREST
            )
        )
        # raise ValueError("image_rgb and prediction_mask must have the same
        # height and width.")

    threshold = cfg.thresholds.metric

    # Binarize the prediction mask using the threshold
    binarized_prediction_mask = (prediction_mask > threshold).astype(np.uint8)

    # Create a colormap for the mask
    # Get the specified colormap
    original_cmap = plt.cm.get_cmap(cmap_name)
    # Create a new colormap where 0 is transparent
    custom_cmap_colors = original_cmap(np.arange(original_cmap.N))
    custom_cmap_colors[0, -1] = (
        0  # Set alpha of the color for value 0 to 0 (transparent)
    )
    custom_cmap = ListedColormap(custom_cmap_colors)

    # Create the figure and axis
    fig, ax = plt.subplots(
        figsize=(image_rgb.shape[1] / 100, image_rgb.shape[0] / 100), dpi=100
    )
    ax.imshow(image_rgb)

    # Overlay the binarized mask
    # Use the binarized mask for the overlay
    ax.imshow(
        binarized_prediction_mask,
        cmap=custom_cmap,
        alpha=alpha,
        interpolation="none",
    )

    ax.axis("off")
    plt.tight_layout(pad=0)

    # Convert plot to image
    fig.canvas.draw()
    overlay_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_image = overlay_image.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close(fig)

    return overlay_image


def get_masked_prediction(
    pred_array: np.ndarray, cfg: DictConfig
) -> np.ma.MaskedArray:
    """Returns a masked array based on a threshold from config."""
    threshold = cfg.thresholds.metric
    return np.ma.masked_where(pred_array < threshold, pred_array)


def visualize_segmentation_results(
    images: list[np.ndarray],
    predictions: list[np.ndarray],
    cfg: DictConfig,
    plot_cfg: PlottingConfig | None = None,
    grid_cfg: GridDisplayConfig | None = None,
):
    if plot_cfg is None:
        plot_cfg = PlottingConfig()
    if grid_cfg is None:
        grid_cfg = GridDisplayConfig()
    num_samples = len(images)
    targets = grid_cfg.targets
    num_cols = (
        cfg.visualization.num_cols
        if targets is not None
        else cfg.visualization.num_cols_no_targets
    )
    fig_size_per_row = grid_cfg.fig_size_per_row
    titles = grid_cfg.titles

    if targets is not None and len(targets) != num_samples:
        raise ValueError("Length of images and targets must match.")
    if len(predictions) != num_samples:
        raise ValueError("Length of images and predictions must match.")

    num_display_cols = (
        cfg.visualization.num_cols
        if targets is not None
        else cfg.visualization.num_cols_no_targets
    )
    num_rows = math.ceil(num_samples / num_cols)

    fig_width = fig_size_per_row[0]
    fig_height = num_rows * fig_size_per_row[1] / (num_display_cols / num_cols)

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_rows, num_display_cols),
        axes_pad=0.3,
        share_all=True,
    )

    threshold = cfg.thresholds.metric

    for i in range(num_samples):
        row = i // num_cols
        col_offset = (i % num_cols) * num_display_cols

        ax_img_idx = row * num_display_cols + col_offset
        ax_pred_idx = row * num_display_cols + col_offset + 1

        img = images[i]
        pred = predictions[i]

        # Original Image
        grid[ax_img_idx].imshow(img)
        grid[ax_img_idx].set_title(
            f"Image {i + 1}"
            if (titles is None or len(titles) <= i)
            else titles[i]
        )
        grid[ax_img_idx].axis("off")

        # Prediction (binarized or masked)
        binarized_pred = (pred > threshold).astype(np.uint8)
        grid[ax_pred_idx].imshow(binarized_pred, cmap="gray")
        grid[ax_pred_idx].set_title(f"Prediction {i + 1}")
        grid[ax_pred_idx].axis("off")

        if targets is not None:
            ax_target_idx = row * num_display_cols + col_offset + 2
            target = targets[i]
            grid[ax_target_idx].imshow(target, cmap="gray")
            grid[ax_target_idx].set_title(f"Target {i + 1}")
            grid[ax_target_idx].axis("off")

    # Remove any unused axes
    for j in range(num_samples * num_display_cols, len(grid)):
        grid[j].axis("off")

    plt.tight_layout()
    if plot_cfg.save_path:
        plt.savefig(plot_cfg.save_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved visualization to {plot_cfg.save_path}")
    if plot_cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig
