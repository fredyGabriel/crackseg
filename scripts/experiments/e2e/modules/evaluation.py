"""Evaluation functions for end-to-end pipeline testing."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataclasses import EvaluationArgs

NO_CHANNEL_DIM = 3


def _evaluate_model_on_test_set(
    args: EvaluationArgs, test_loader: DataLoader[Any]
) -> tuple[float, dict[str, float], Any, Any, Any]:
    """Evaluates the model on the test set and returns metrics and sample
    predictions."""
    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestEvaluation") if get_logger else None

    # Load the best model
    checkpoint_path = args.checkpoints_dir / args.cfg_model_to_load
    if logger:
        logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    args.model.load_state_dict(checkpoint["model_state_dict"])
    args.model.eval()

    test_loss = 0.0
    test_metrics = dict.fromkeys(args.metrics_dict.keys(), 0.0)
    sample_images = []
    sample_masks = []
    sample_preds = []

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

            # Calculate metrics
            for k, metric_fn in args.metrics_dict.items():
                test_metrics[k] += metric_fn(outputs, targets).item()

            # Store samples for visualization
            if batch_idx == 0:  # Store first batch
                sample_images = inputs[:4].cpu()
                sample_masks = targets[:4].cpu()
                sample_preds = torch.sigmoid(outputs[:4]).cpu()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_metrics = {
        k: v / len(test_loader) for k, v in test_metrics.items()
    }

    if logger:
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
        for k, v in avg_test_metrics.items():
            logger.info(f"Test {k}: {v:.4f}")

    return (
        avg_test_loss,
        avg_test_metrics,
        sample_images,
        sample_masks,
        sample_preds,
    )
