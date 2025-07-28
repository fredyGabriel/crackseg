"""Metrics calculation for crack segmentation evaluation."""

import logging

import torch

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate segmentation metrics for crack detection."""

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new predictions and targets.

        Args:
            predictions: Binary prediction tensor (B, 1, H, W)
            targets: Binary target tensor (B, 1, H, W)
        """
        # Ensure tensors are binary
        pred_binary = (predictions > 0.5).float()
        target_binary = (targets > 0.5).float()

        # Flatten tensors for calculation
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)

        # Calculate confusion matrix components
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

        # Accumulate
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self) -> dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary containing computed metrics
        """
        eps = 1e-8

        # Calculate metrics
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (
            self.tp + self.tn + self.fp + self.fn + eps
        )

        # Calculate IoU
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        iou = intersection / (union + eps)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "iou": float(iou),
        }

    def calculate_single_batch(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """
        Calculate metrics for a single batch.

        Args:
            predictions: Binary prediction tensor
            targets: Binary target tensor

        Returns:
            Dictionary containing metrics for this batch
        """
        self.reset()
        self.update(predictions, targets)
        return self.compute()

    def calculate_iou(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        Calculate IoU for a single batch.

        Args:
            predictions: Binary prediction tensor
            targets: Binary target tensor

        Returns:
            IoU score
        """
        # Ensure tensors are binary
        pred_binary = (predictions > 0.5).float()
        target_binary = (targets > 0.5).float()

        # Calculate intersection and union
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-8)
        return float(iou.item())
