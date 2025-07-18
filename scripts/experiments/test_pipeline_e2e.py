#!/usr/bin/env python
"""
End-to-end test script to verify the full training pipeline.

This script performs a complete pipeline test:
1. Loads a small synthetic dataset
2. Configures a reduced UNet model
3. Trains for a few epochs
4. Saves checkpoints at intervals
5. Loads the model from checkpoint
6. Evaluates on the test set
7. Generates a results report
"""

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to Python path to allow direct execution of the script
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from crackseg.data.factory import (  # noqa: E402
    create_dataloaders_from_config,
)
from crackseg.model.factory import create_unet  # noqa: E402
from crackseg.training.factory import create_lr_scheduler  # noqa: E402
from crackseg.utils.factory import get_loss_fn, get_optimizer  # noqa: E402
from crackseg.utils.logging import (  # noqa: E402
    ExperimentLogger,
    get_logger,
)

# Logging configuration
logger = get_logger("TestPipelineE2E")

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import yaml
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from crackseg.training.metrics import F1Score, IoUScore
from crackseg.utils import (
    load_checkpoint,
    save_checkpoint,
    set_random_seeds,
)
from crackseg.utils.checkpointing import CheckpointSaveConfig

NO_CHANNEL_DIM = 3


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
    experiment_logger: ExperimentLogger
    checkpoints_dir: Path


@dataclass
class EvaluationArgs:
    """Arguments for the model evaluation function."""

    model: torch.nn.Module
    loss_fn: torch.nn.Module
    metrics_dict: dict[str, Any]
    device: torch.device
    experiment_logger: ExperimentLogger
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


def create_mini_config():
    """Create a minimal config for testing."""
    config = {
        "data": {
            "data_root": "data",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": [256, 256],
            "batch_size": 4,
            "num_workers": 2,
            "seed": 42,
            "in_memory_cache": False,
            "transforms": {
                "train": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
                "val": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
                "test": [
                    {
                        "name": "Resize",
                        "params": {"height": 256, "width": 256},
                    },
                    {
                        "name": "Normalize",
                        "params": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                    },
                ],
            },
        },
        "model": {
            "_target_": "src.model.unet.BaseUNet",
            "encoder": {
                "type": "CNNEncoder",
                "in_channels": 3,
                "init_features": 16,  # Fixed: base_channels -> init_features
                "depth": 3,  # Reduced for test
            },
            "bottleneck": {
                "type": "CNNBottleneckBlock",  # Fixed: CNNBottleneckBlock
                "in_channels": 64,  # Adapted to encoder
                "out_channels": 128,  # Reduced for test
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 128,  # Adapted to bottleneck
                "skip_channels_list": [16, 32, 64],
                "out_channels": 1,
                "depth": 3,  # Must match encoder
            },
            "final_activation": {"_target_": "torch.nn.Sigmoid"},
        },
        "training": {
            "epochs": 2,  # Few epochs for quick test
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.5,
            },
            "loss": {
                "_target_": "src.training.losses.BCEDiceLoss",
                "bce_weight": 0.5,
                "dice_weight": 0.5,
            },
            "checkpoints": {
                "save_interval_epochs": 1,
                "save_best": {
                    "enabled": True,
                    "monitor_metric": "val_iou",
                    "monitor_mode": "max",
                },
                "save_last": True,
            },
            "amp_enabled": False,  # Disabled for simplicity
        },
        "evaluation": {
            "metrics": {
                "dice": {"_target_": "src.training.metrics.F1Score"},
                "iou": {"_target_": "src.training.metrics.IoUScore"},
            }
        },
        "random_seed": 42,
        "require_cuda": False,
        "log_level": "INFO",
        "log_to_file": True,
    }
    return OmegaConf.create(config)


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


def create_experiment_dir():
    """Create experiment directory."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(project_root, "outputs", "e2e_test", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)

    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def visualize_results(
    images: list[torch.Tensor],
    masks: list[torch.Tensor],
    predictions: list[torch.Tensor],
    output_path: str,
) -> None:
    """Save visualizations of predictions vs ground truth."""
    num_samples = min(4, len(images))
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

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


def create_synthetic_dataset():
    """Create a synthetic dataset for end-to-end tests."""
    # Create synthetic data (random 16x16 images)
    num_samples = 20
    image_size = 16

    # Create random tensors for images and masks
    images = torch.rand(num_samples, 3, image_size, image_size)
    masks = torch.randint(
        0, 2, (num_samples, 1, image_size, image_size)
    ).float()

    # Create dataset and split into train/val/test
    dataset = TensorDataset(images, masks)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def get_metrics_from_cfg(metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    """Get metrics from config."""
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


def _setup_experiment_resources(
    base_cfg_callable: Callable[[], Any],
) -> tuple[Any, Path, torch.device, ExperimentLogger, Path, Path, Path]:
    """
    Handles creation of config, experiment dirs, logger, and seed/device setup.
    """
    cfg = OmegaConf.create(base_cfg_callable())
    # Forzamos el tipo a DictConfig para ExperimentLogger
    # (OmegaConf.create puede retornar ListConfig,
    # pero aquí debe ser DictConfig)
    cfg = cast(DictConfig, cfg)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Ensure exp_dir is absolute, joining with project_root or a robust base
    # path
    Path(__file__).parent.absolute()
    exp_dir_base = Path(project_root) / "outputs" / "e2e_test"
    exp_dir = exp_dir_base / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "config.yaml"
    save_config(cfg, str(config_path))  # save_config expects string path

    checkpoints_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    vis_dir = exp_dir / "visualizations"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    experiment_logger = ExperimentLogger(
        log_dir=str(exp_dir),  # ExperimentLogger might expect string
        experiment_name="e2e_test",
        config=cfg,
        log_level="INFO",
        log_to_file=True,
    )
    set_random_seeds(cfg.random_seed)
    # OmegaConf get workaround for require_cuda
    require_cuda = cfg.require_cuda if hasattr(cfg, "require_cuda") else False
    device = torch.device(
        "cuda" if torch.cuda.is_available() and require_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")
    return (
        cfg,
        exp_dir,
        device,
        experiment_logger,
        checkpoints_dir,
        metrics_dir,
        vis_dir,
    )


def _prepare_dataloaders(
    cfg_data: Any,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Loads and returns train, val, and test dataloaders."""
    logger.info("Loading data...")
    dataloaders = create_dataloaders_from_config(
        data_config=cfg_data,
        transform_config=cfg_data.get("transforms", {}),
        # Pass the base data_config for dataloader params
        dataloader_config=cfg_data,
    )
    train_loader = dataloaders["train"]["dataloader"]
    val_loader = dataloaders["val"]["dataloader"]
    test_loader = dataloaders["test"]["dataloader"]

    def get_dataset_len(loader: Any) -> int:
        if isinstance(loader, DataLoader):
            # mypy: asegúrate de que loader.dataset es Sized
            from collections.abc import Sized

            dataset = loader.dataset
            if isinstance(dataset, Sized):
                return len(dataset)
            return -1
        elif hasattr(loader, "__len__"):
            return len(loader)
        return -1

    logger.info(
        f"Data loaded: {get_dataset_len(train_loader)} train, "
        f"{get_dataset_len(val_loader)} val, "
        f"{get_dataset_len(test_loader)} test"
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    return train_loader, val_loader, test_loader


def _initialize_training_components(cfg: Any, device: torch.device) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    Any,
    dict[str, Any],
    Any,
]:
    """
    Initializes model, loss, optimizer, scheduler, metrics, and AMP scaler.
    """
    logger.info("Creating model and training components...")
    model = create_unet(cfg.model).to(device)
    loss_fn = get_loss_fn(cfg.training.loss)
    # Si loss_fn no es nn.Module, lo envuelvo en nn.Module para
    # cumplir el tipado estricto
    if not isinstance(loss_fn, torch.nn.Module):

        class LossWrapper(torch.nn.Module):
            def __init__(self, fn: Callable[..., Any]) -> None:
                super().__init__()
                self.fn = fn

            def forward(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
                return self.fn(*args, **kwargs)

        loss_fn = LossWrapper(loss_fn)
    optimizer = get_optimizer(list(model.parameters()), cfg.training.optimizer)
    lr_scheduler = create_lr_scheduler(optimizer, cfg.training.scheduler)
    metrics_dict = get_metrics_from_cfg(cfg.evaluation.metrics)
    use_amp = cfg.training.get("amp_enabled", False)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    logger.info(f"AMP Enabled: {scaler is not None}")
    logger.info("Training components initialized.")
    return model, loss_fn, optimizer, lr_scheduler, metrics_dict, scaler


def _run_train_epoch(
    args: TrainingRunArgs, train_loader: DataLoader[Any]
) -> tuple[float, dict[str, float]]:
    """Runs a single training epoch and returns loss and metrics."""
    args.model.train()
    epoch_loss = 0.0
    epoch_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)

    for batch_idx, batch in enumerate(train_loader):
        inputs, targets = (
            batch["image"].to(args.device),
            batch["mask"].to(args.device),
        )
        if inputs.shape[-1] == NO_CHANNEL_DIM:
            inputs = inputs.permute(0, 3, 1, 2)
        if len(targets.shape) == NO_CHANNEL_DIM:
            targets = targets.unsqueeze(1)

        args.optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(args.scaler is not None)):
            outputs = args.model(inputs)
            loss = args.loss_fn(outputs, targets)

        if args.scaler:
            args.scaler.scale(loss).backward()
            args.scaler.step(args.optimizer)
            args.scaler.update()
        else:
            loss.backward()
            args.optimizer.step()

        epoch_loss += loss.item()
        with torch.no_grad():
            for k, metric_fn in args.metrics_dict.items():
                epoch_metrics[k] += metric_fn(outputs, targets).item()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            logger.info(
                f"Train Batch: {batch_idx + 1}/{len(train_loader)}, Loss: "
                f"{loss.item():.4f}"
            )

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_metrics = {
        k: v / len(train_loader) for k, v in epoch_metrics.items()
    }
    return avg_epoch_loss, avg_epoch_metrics


def _run_val_epoch(
    args: TrainingRunArgs, val_loader: DataLoader[Any]
) -> tuple[float, dict[str, float]]:
    """Runs a single validation epoch and returns loss and metrics."""
    args.model.eval()
    epoch_loss = 0.0
    epoch_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = (
                batch["image"].to(args.device),
                batch["mask"].to(args.device),
            )
            if inputs.shape[-1] == NO_CHANNEL_DIM:
                inputs = inputs.permute(0, 3, 1, 2)
            if len(targets.shape) == NO_CHANNEL_DIM:
                targets = targets.unsqueeze(1)
            outputs = args.model(inputs)
            loss = args.loss_fn(outputs, targets)
            epoch_loss += loss.item()
            for k, metric_fn in args.metrics_dict.items():
                epoch_metrics[k] += metric_fn(outputs, targets).item()

    avg_epoch_loss = epoch_loss / len(val_loader)
    avg_epoch_metrics = {
        k: v / len(val_loader) for k, v in epoch_metrics.items()
    }
    return avg_epoch_loss, avg_epoch_metrics


def _execute_training_and_validation(
    args: TrainingRunArgs,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
) -> tuple[float, float, dict[str, float], float, dict[str, float]]:
    """Runs the main training and validation loop for all epochs."""
    logger.info("Starting training loop...")
    best_metric_value = 0.0
    # Initialize return values for cases where training loop might not run
    # (e.g. 0 epochs)
    # These should be updated with actual final epoch values inside the loop.
    final_train_loss = float("nan")
    final_train_metrics = {k: float("nan") for k in args.metrics_dict}
    final_val_loss = float("nan")
    final_val_metrics = {k: float("nan") for k in args.metrics_dict}

    for epoch in range(args.cfg_training.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.cfg_training.epochs}")

        # Training phase for one epoch
        final_train_loss, final_train_metrics = _run_train_epoch(
            args, train_loader
        )
        logger.info(
            f"Train Loss: {final_train_loss:.4f}, Metrics: "
            f"{final_train_metrics}"
        )
        args.experiment_logger.log_scalar(
            "train/loss", final_train_loss, epoch
        )
        for k, v in final_train_metrics.items():
            args.experiment_logger.log_scalar(f"train/{k}", v, epoch)

        # Validation phase for one epoch
        final_val_loss, final_val_metrics = _run_val_epoch(args, val_loader)
        logger.info(
            f"Validation Loss: {final_val_loss:.4f}, Metrics: "
            f"{final_val_metrics}"
        )
        args.experiment_logger.log_scalar("val/loss", final_val_loss, epoch)
        for k, v in final_val_metrics.items():
            args.experiment_logger.log_scalar(f"val/{k}", v, epoch)

        # Learning rate scheduler step
        if args.lr_scheduler:
            if isinstance(
                args.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                args.lr_scheduler.step(final_val_loss)
            else:
                args.lr_scheduler.step()

        current_monitor_metric = final_val_metrics.get(
            args.cfg_training.checkpoints.save_best.monitor_metric,
            (
                float("-inf")
                if args.cfg_training.checkpoints.save_best.monitor_mode
                == "max"
                else float("inf")
            ),
        )
        is_best = (
            args.cfg_training.checkpoints.save_best.monitor_mode == "max"
            and current_monitor_metric > best_metric_value
        ) or (
            args.cfg_training.checkpoints.save_best.monitor_mode == "min"
            and current_monitor_metric < best_metric_value
        )
        if is_best:
            best_metric_value = current_monitor_metric
            logger.info(
                f"New best "
                f"{args.cfg_training.checkpoints.save_best.monitor_metric}: "
                f"{best_metric_value:.4f}"
            )
            # Crear configuración de guardado de checkpoint
            checkpoint_config = CheckpointSaveConfig(
                checkpoint_dir=args.checkpoints_dir,
                filename="model_best.pth.tar",
                additional_data={
                    "scheduler_state_dict": (
                        args.lr_scheduler.state_dict()
                        if args.lr_scheduler
                        else None
                    ),
                    "best_metric_value": best_metric_value,
                    "config": OmegaConf.to_container(
                        args.cfg_training, resolve=True
                    ),
                },
            )

            save_checkpoint(
                args.model, args.optimizer, epoch, checkpoint_config
            )

        if (
            args.cfg_training.checkpoints.save_interval_epochs
            and (epoch + 1)
            % args.cfg_training.checkpoints.save_interval_epochs
            == 0
        ):
            # Crear configuración de guardado de checkpoint para intervalos
            checkpoint_config = CheckpointSaveConfig(
                checkpoint_dir=args.checkpoints_dir,
                filename=f"checkpoint_epoch_{epoch + 1}.pth.tar",
                additional_data={
                    "scheduler_state_dict": (
                        args.lr_scheduler.state_dict()
                        if args.lr_scheduler
                        else None
                    ),
                    "best_metric_value": best_metric_value,
                    "config": OmegaConf.to_container(
                        args.cfg_training, resolve=True
                    ),
                },
            )

            save_checkpoint(
                args.model, args.optimizer, epoch, checkpoint_config
            )
        if args.cfg_training.checkpoints.save_last:
            # Crear configuración de guardado de checkpoint para el último
            checkpoint_config = CheckpointSaveConfig(
                checkpoint_dir=args.checkpoints_dir,
                filename="checkpoint_last.pth.tar",
                additional_data={
                    "scheduler_state_dict": (
                        args.lr_scheduler.state_dict()
                        if args.lr_scheduler
                        else None
                    ),
                    "best_metric_value": best_metric_value,
                    "config": OmegaConf.to_container(
                        args.cfg_training, resolve=True
                    ),
                },
            )

            save_checkpoint(
                args.model, args.optimizer, epoch, checkpoint_config
            )

    logger.info("Training finished.")
    return (
        best_metric_value,
        final_train_loss,
        final_train_metrics,
        final_val_loss,
        final_val_metrics,
    )


def _evaluate_model_on_test_set(
    args: EvaluationArgs,
    test_loader: DataLoader[Any],
) -> tuple[float, dict[str, float], str]:
    """Loads the specified model and evaluates it on the test set."""
    logger.info(
        f"Loading model '{args.cfg_model_to_load}' for test evaluation..."
    )
    model_path = args.checkpoints_dir / args.cfg_model_to_load

    if model_path.exists():
        checkpoint_data = load_checkpoint(
            model=args.model,
            checkpoint_path=str(model_path),
            device=args.device,
        )
        logger.info(f"Model loaded from: {model_path}")
        logger.info(
            f"Model from epoch: {checkpoint_data.get('epoch', 'unknown')}"
        )
    else:
        logger.warning(
            f"Checkpoint not found at {model_path}. Evaluating with current "
            "model state."
        )

    args.model.eval()
    test_loss = 0.0
    current_test_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)
    sample_images, sample_masks, sample_preds = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = (
                batch["image"].to(args.device),
                batch["mask"].to(args.device),
            )
            if inputs.shape[-1] == NO_CHANNEL_DIM:
                inputs = inputs.permute(0, 3, 1, 2)
            if len(targets.shape) == NO_CHANNEL_DIM:
                targets = targets.unsqueeze(1)

            outputs = args.model(inputs)
            loss = args.loss_fn(outputs, targets)
            test_loss += loss.item()
            for k, metric_fn in args.metrics_dict.items():
                current_test_metrics[k] += metric_fn(outputs, targets).item()

            if batch_idx == 0:  # Save samples only from the first batch
                sample_images.extend(inputs.cpu())
                sample_masks.extend(targets.cpu())
                sample_preds.extend(outputs.cpu())

    final_test_loss = test_loss / len(test_loader)
    final_test_metrics = {
        k: v / len(test_loader) for k, v in current_test_metrics.items()
    }

    logger.info(
        f"Test Loss: {final_test_loss:.4f}, Metrics: {final_test_metrics}"
    )
    args.experiment_logger.log_scalar(
        "test/loss", final_test_loss, 0
    )  # epoch 0 for test results
    for k, v in final_test_metrics.items():
        args.experiment_logger.log_scalar(f"test/{k}", v, 0)

    if sample_images:
        logger.info("Generating visualizations...")
        visualize_results(
            sample_images[:4],
            sample_masks[:4],
            sample_preds[:4],
            str(args.vis_dir / "test_predictions.png"),
        )
    return final_test_loss, final_test_metrics, str(model_path)


def _finalize_and_save_results(args: FinalResultsData):
    """Compiles and saves final training and evaluation results."""
    final_results_data = {
        "train_summary": {
            "loss": args.final_train_loss,
            "metrics": args.final_train_metrics,
        },
        "val_summary": {
            "loss": args.final_val_loss,
            "metrics": args.final_val_metrics,
        },
        "test_summary": {"loss": args.test_loss, "metrics": args.test_metrics},
        "training_epochs": args.epochs,
        "best_validation_metric_value": args.best_metric_val,
        "loaded_checkpoint_for_test": args.loaded_checkpoint_path,
        "experiment_directory": str(args.exp_dir),
    }
    results_path = args.metrics_dir / "final_e2e_results.yaml"
    results_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure metrics_dir exists
    with open(results_path, "w") as f:
        yaml.dump(final_results_data, f, default_flow_style=False)
    logger.info(f"Final results saved to {results_path}")
    return final_results_data


def run_e2e_test() -> tuple[Any, Any]:
    """Run end-to-end test of the full pipeline."""
    exp_dir_final, results_final = None, None  # Initialize for finally block
    experiment_logger_instance = None  # Initialize for finally block

    try:
        # 1. Setup experiment (config, dirs, logger, device)
        (
            cfg,
            exp_dir_final,
            device,
            experiment_logger_instance,
            checkpoints_dir,
            metrics_dir,
            vis_dir,
        ) = _setup_experiment_resources(create_mini_config)

        # 2. Prepare Dataloaders
        train_loader, val_loader, test_loader = _prepare_dataloaders(cfg.data)

        # 3. Initialize Model and Training Components
        (model, loss_fn, optimizer, lr_scheduler, metrics_dict, scaler) = (
            _initialize_training_components(cfg, device)
        )

        # Create TrainingRunArgs instance
        training_run_args = TrainingRunArgs(
            cfg_training=cfg.training,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics_dict=metrics_dict,
            scaler=scaler,
            device=device,
            experiment_logger=experiment_logger_instance,
            checkpoints_dir=checkpoints_dir,
        )

        # 4. Execute Training and Validation Loop
        (best_val_metric, train_loss, train_metrics, val_loss, val_metrics) = (
            _execute_training_and_validation(
                training_run_args,  # Pass the single args object
                train_loader,
                val_loader,
            )
        )

        # Prepare arguments for evaluation
        evaluation_args = EvaluationArgs(
            model=model,
            loss_fn=loss_fn,
            metrics_dict=metrics_dict,
            device=device,
            experiment_logger=experiment_logger_instance,
            vis_dir=vis_dir,
            checkpoints_dir=checkpoints_dir,
            # cfg_model_to_load will use its default "model_best.pth.tar"
        )

        # 5. Evaluate Model on Test Set (loads best model by default)
        test_loss_val, test_metrics_vals, loaded_model_path = (
            _evaluate_model_on_test_set(
                evaluation_args,  # Pass the single args object
                test_loader,
            )
        )

        # Prepare arguments for finalizing results
        results_args = FinalResultsData(
            exp_dir=exp_dir_final,
            metrics_dir=metrics_dir,
            final_train_loss=train_loss,
            final_train_metrics=train_metrics,
            final_val_loss=val_loss,
            final_val_metrics=val_metrics,
            test_loss=test_loss_val,
            test_metrics=test_metrics_vals,
            epochs=cfg.training.epochs,
            best_metric_val=best_val_metric,
            loaded_checkpoint_path=loaded_model_path,
        )

        # 6. Finalize and Save Results
        results_final = _finalize_and_save_results(results_args)

        logger.info("End-to-end test completed successfully.")

    except Exception as e:
        logger.exception(f"Error during end-to-end test: {str(e)}")
        raise
    finally:
        if experiment_logger_instance:
            experiment_logger_instance.close()
        logger.info("Resources released.")

    return exp_dir_final, results_final


if __name__ == "__main__":
    experiment_dir, results = run_e2e_test()
    print(f"\n{'=' * 80}")
    print(" End-to-End Test Completed")
    print(f" Experiment directory: {experiment_dir}")
    # Acceso seguro a los resultados
    if results and isinstance(results, dict):
        test_metrics = results.get("test_summary", {}).get("metrics", {})
        print(f" Test metrics: {test_metrics}")
    print(f"{'=' * 80}")
