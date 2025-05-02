"""
Loss functions for crack segmentation.

This module provides loss functions specifically designed for binary
segmentation of cracks, including BCE and Dice loss implementations.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class SegmentationLoss(nn.Module):
    """Base class for segmentation loss functions."""

    def __init__(self):
        """Initialize the base loss class."""
        super().__init__()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss calculation.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Calculated loss value
        """
        raise NotImplementedError


class BCELoss(SegmentationLoss):
    """Binary Cross Entropy loss for binary segmentation."""

    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Initialize BCE loss.

        Args:
            weight: Optional weight for unbalanced datasets
            reduction: Specifies the reduction to apply to the output:
                       'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate BCE loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            BCE loss value
        """
        return self.bce(pred, target)


class DiceLoss(SegmentationLoss):
    """Dice loss for binary segmentation with optional smoothing."""

    def __init__(self, smooth: float = 1.0, sigmoid: bool = True):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor to prevent division by zero
            sigmoid: Whether to apply sigmoid to predictions
        """
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Dice loss value
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        # Add small epsilon to prevent division by zero
        eps = 1e-6 if self.smooth == 0 else self.smooth

        intersection = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()

        # Handle edge cases
        if target_sum == 0:
            # If target is all zeros, pred should also be all zeros
            return torch.clamp(pred_sum / (pred_sum + eps), max=1.0)

        if pred_sum == 0 and target_sum == 0:
            return torch.tensor(0.0, device=pred.device)

        # Calculate Dice coefficient
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

        # Ensure loss is in [0, 1] range and high for wrong predictions
        return torch.clamp(1.0 - dice, min=0.0, max=1.0)


class FocalLoss(SegmentationLoss):
    """
    Focal Loss for dealing with class imbalance in segmentation tasks.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the ground truth class.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        sigmoid: bool = True
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for the rare class (default: 0.25)
            gamma: Focusing parameter to down-weight easy examples
                (default: 2.0)
            sigmoid: Whether to apply sigmoid to predictions (default: True)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = sigmoid
        self.eps = 1e-6  # For numerical stability

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Focal loss value
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        # Ensure numerical stability
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Apply focal modulation
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = torch.where(
                target == 1,
                self.alpha * torch.ones_like(target),
                (1 - self.alpha) * torch.ones_like(target)
            )
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * bce

        # Return mean loss
        return loss.mean()


class CombinedLoss(SegmentationLoss):
    """
    Combined loss function that applies multiple loss functions with weights.
    """

    def __init__(self, losses: List[SegmentationLoss],
                 weights: Optional[List[float]] = None):
        """
        Initialize combined loss function.

        Args:
            losses: List of loss functions to combine
            weights: Optional list of weights for each loss function.
                    If None, equal weights are used.
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)

        if weights is None:
            weights = [1.0] * len(losses)

        if len(weights) != len(losses):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of losses ({len(losses)})"
            )

        # Normalize weights to sum to 1.0
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(pred, target)
            total_loss += weight * loss_value

        return total_loss


class BCEDiceLoss(SegmentationLoss):
    """
    Combined BCE and Dice loss for binary segmentation.
    This is a common combination for segmentation tasks.
    """

    def __init__(self,
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 smooth: float = 1.0):
        """
        Initialize BCE+Dice loss.

        Args:
            bce_weight: Weight for BCE loss component (default: 0.5)
            dice_weight: Weight for Dice loss component (default: 0.5)
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.combined = CombinedLoss(
            losses=[BCELoss(), DiceLoss(smooth=smooth)],
            weights=[bce_weight, dice_weight]
        )

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate BCE+Dice loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)

        Returns:
            BCE+Dice loss value
        """
        return self.combined(pred, target)
