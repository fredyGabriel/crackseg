"""Environment setup for evaluation CLI.

This module provides functions for setting up the evaluation environment
including device configuration, output directory setup, and checkpoint validation.
"""

import argparse
import os
from pathlib import Path

import torch

from crackseg.utils import get_device, set_random_seeds
from crackseg.utils.logging import get_logger

# Configure logger
log = get_logger("evaluation")


def setup_evaluation_environment(
    args: argparse.Namespace,
) -> tuple[torch.device, Path, list[str]]:
    """Sets up the evaluation environment: seed, device, output dir, checkpoints.

    Args:
        args: Command line arguments containing evaluation parameters.

    Returns:
        tuple containing:
            - torch.device: Device to use for evaluation
            - Path: Output directory path
            - list[str]: List of checkpoint paths

    Raises:
        FileNotFoundError: If checkpoint files are not found.
    """
    set_random_seeds(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    log.info(f"Using device: {device}")

    output_dir_str = setup_output_directory(args.output_dir)
    output_dir = Path(output_dir_str)
    log.info(f"Output directory set to: {output_dir}")

    checkpoint_paths = (
        [cp.strip() for cp in args.checkpoint.split(",")]
        if args.ensemble
        else [args.checkpoint]
    )
    for cp_path in checkpoint_paths:
        if not os.path.exists(cp_path):
            raise FileNotFoundError(f"Checkpoint file not found: {cp_path}")
        log.info(f"Found checkpoint: {cp_path}")
    return device, output_dir, checkpoint_paths


def setup_output_directory(output_dir: str | None) -> str:
    """Setup output directory for evaluation results.

    Args:
        output_dir: Output directory path or None for default.

    Returns:
        str: Output directory path.
    """
    if output_dir is None:
        output_dir = "outputs/evaluation"

    os.makedirs(output_dir, exist_ok=True)
    return output_dir
