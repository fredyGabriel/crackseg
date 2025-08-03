"""Component preparation for evaluation CLI.

This module provides functions for preparing evaluation components including
data loaders, metrics, and experiment logging.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from crackseg.evaluation.data import get_evaluation_dataloader
from crackseg.evaluation.loading import load_model_from_checkpoint
from crackseg.utils.factory import get_metrics_from_cfg
from crackseg.utils.logging import get_logger
from crackseg.utils.logging.experiment import ExperimentLogger

# Configure logger
log = get_logger("evaluation")


def get_evaluation_components(
    cfg: DictConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[DataLoader[Any], dict[str, Any], ExperimentLogger]:
    """Prepares evaluation components: dataloader, metrics, and logger.

    Args:
        cfg: Configuration object.
        args: Command line arguments.
        output_dir: Output directory for results.

    Returns:
        tuple containing:
            - DataLoader: Test dataloader
            - dict: Metrics dictionary
            - ExperimentLogger: Logger for experiment tracking
    """
    # Prepare test dataloader
    log.info("Preparing test dataloader...")
    test_loader = get_evaluation_dataloader(cfg.data, args.test_data_path)
    log.info(f"Test dataloader prepared with {len(test_loader)} batches")

    # Prepare metrics
    log.info("Preparing evaluation metrics...")
    metrics_dict = get_metrics_from_cfg(cfg.evaluation.metrics)
    log.info(f"Prepared {len(metrics_dict)} metrics for evaluation")

    # Setup experiment logger
    log.info("Setting up experiment logger...")
    experiment_logger = ExperimentLogger(
        log_dir=output_dir / "logs",
        experiment_name=args.experiment_name or "evaluation",
    )
    log.info("Experiment logger setup complete")

    return test_loader, metrics_dict, experiment_logger


def load_models_for_evaluation(
    checkpoint_paths: list[str], device: torch.device
) -> list[torch.nn.Module]:
    """Load models from checkpoints for evaluation.

    Args:
        checkpoint_paths: List of checkpoint paths.
        device: Device to load models on.

    Returns:
        list[torch.nn.Module]: List of loaded models.
    """
    models = []
    for checkpoint_path in checkpoint_paths:
        log.info(f"Loading model from: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, device)
        models.append(model)

    log.info(f"Loaded {len(models)} models for evaluation")
    return models
