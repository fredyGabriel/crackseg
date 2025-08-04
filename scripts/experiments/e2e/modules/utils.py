"""Utility functions for end-to-end pipeline testing."""

import os
import time
from typing import Any

import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf

from crackseg.utils.logging import get_logger

logger = get_logger("E2ETestUtils")


def save_config(config: Any, path: str) -> None:
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            OmegaConf.to_container(config, resolve=True),
            f,
            default_flow_style=False,
        )
    logger.info(f"Config saved to {path}")


def create_experiment_dir(project_root: str) -> str:
    """Create experiment directory."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(project_root, "artifacts", "e2e_test", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)

    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def visualize_results(
    images: list[Any],
    masks: list[Any],
    predictions: list[Any],
    output_path: str,
) -> None:
    """Save visualizations of predictions vs ground truth."""
    num_samples = min(4, len(images))
    _, axs = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        # Convert tensors to numpy and prepare for visualization
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize

        mask = masks[i].cpu().squeeze().numpy()
        pred = predictions[i].cpu().squeeze().numpy()

        # Plot image, mask, and prediction
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Original image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(mask, cmap="gray")
        axs[i, 1].set_title("Ground truth mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred, cmap="gray")
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def get_metrics_from_cfg(metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    """Get metrics from config."""
    from crackseg.training.metrics import F1Score, IoUScore

    metrics: dict[str, Any] = {}
    for name, metric_config in metrics_cfg.items():
        if metric_config.get("_target_", "").endswith("F1Score"):
            metrics[name] = F1Score()
        elif metric_config.get("_target_", "").endswith("IoUScore"):
            metrics[name] = IoUScore()
    # If no metrics defined, use defaults
    if not metrics:
        metrics = {"dice": F1Score(), "iou": IoUScore()}
    return metrics
