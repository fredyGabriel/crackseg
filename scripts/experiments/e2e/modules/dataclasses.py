"""Data classes for end-to-end pipeline testing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingRunArgs:
    """Arguments for the training and validation execution loop."""

    cfg_training: Any  # OmegaConf subtree
    model: torch.nn.Module
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Any  # e.g., torch.optim.lr_scheduler._LRScheduler
    metrics_dict: dict[str, Any]
    scaler: torch.cuda.amp.GradScaler | None
    device: torch.device
    experiment_logger: Any  # ExperimentLogger
    checkpoints_dir: Path


@dataclass
class EvaluationArgs:
    """Arguments for the model evaluation function."""

    model: torch.nn.Module
    loss_fn: torch.nn.Module
    metrics_dict: dict[str, Any]
    device: torch.device
    experiment_logger: Any  # ExperimentLogger
    vis_dir: Path
    checkpoints_dir: Path
    cfg_model_to_load: str = "model_best.pth.tar"


@dataclass
class FinalResultsData:
    """Arguments for finalizing and saving results."""

    exp_dir: Path
    metrics_dir: Path
    final_train_loss: float
    final_train_metrics: dict[str, float]
    final_val_loss: float
    final_val_metrics: dict[str, float]
    test_loss: float
    test_metrics: dict[str, float]
    epochs: int
    best_metric_val: float
    loaded_checkpoint_path: str
