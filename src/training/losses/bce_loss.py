# src/training/losses/bce_loss.py

import torch
from torch import nn

from src.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss  # Import base class


@loss_registry.register(
    name="bce_loss", tags=["segmentation", "binary", "cross_entropy"]
)
class BCELoss(SegmentationLoss):
    """Binary Cross Entropy loss for binary segmentation."""

    def __init__(
        self, weight: torch.Tensor | None = None, reduction: str = "mean"
    ):
        """
        Initialize BCE loss.

        Args:
            weight: Optional weight for unbalanced datasets
            reduction: Specifies the reduction to apply to the output:
                       'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        # Use BCEWithLogitsLoss for numerical stability and to handle sigmoid
        # internally
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate BCE loss between predicted logits and target masks.

        Args:
            pred: Predicted segmentation map (logits) (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            BCE loss value
        """
        return self.bce(pred, target)
