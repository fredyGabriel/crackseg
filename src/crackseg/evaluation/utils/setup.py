import argparse
import os
from datetime import datetime

from crackseg.utils.logging import get_logger

log = get_logger("evaluation.setup")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (if not stored in checkpoint)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to test dataset (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results \
(default: ./artifacts/evaluation/TIMESTAMP)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable ensemble evaluation with multiple checkpoints",
    )
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    return parser.parse_args()


def setup_output_directory(base_dir: str | None = None) -> str:
    """
    Create output directory for evaluation results.

    Args:
        base_dir: Base directory path (if None, uses ./artifacts/evaluation/)

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if base_dir is None:
        base_dir = os.path.join("artifacts", "evaluation")

    output_dir = os.path.join(base_dir, timestamp)

    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    log.info("Evaluation results will be saved to: %s", output_dir)
    return output_dir
