# src/training/losses/base_loss.py
import torch
from torch import nn


class SegmentationLoss(nn.Module):
    """Base class for segmentation loss functions."""

    def __init__(self):
        """Initialize the base loss class."""
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the loss calculation.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Calculated loss value
        """
        raise NotImplementedError
