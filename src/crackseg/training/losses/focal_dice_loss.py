# src/training/losses/focal_dice_loss.py
from dataclasses import dataclass
from typing import Any, cast

import torch

from crackseg.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss
from .combined_loss import CombinedLoss


@dataclass
class FocalDiceLossConfig:
    """Configuration for Focal + Dice Loss combination."""

    focal_weight: float = 0.6
    dice_weight: float = 0.4
    focal_alpha: float = (
        0.25  # Optimized for crack segmentation (<5% positive pixels)
    )
    focal_gamma: float = 2.0  # Focus on hard examples
    focal_reduction: str = "mean"
    dice_smooth: float = 1.0
    dice_sigmoid: bool = True
    dice_eps: float = 1e-6
    total_loss_weight: float = 1.0


@loss_registry.register(
    name="focal_dice_loss",
    tags=["segmentation", "focal", "dice", "combined", "crack"],
    force=True,
)
class FocalDiceLoss(SegmentationLoss):
    """
    Optimized combination of Focal Loss and Dice Loss for crack segmentation.

    This loss function is specifically designed for pavement crack detection
    where class imbalance is extreme (<5% positive pixels) and thin structures
    (1-5 pixels wide) need to be preserved.

    Focal Loss handles the severe class imbalance by focusing on hard examples,
    while Dice Loss optimizes directly for the segmentation metric.
    """

    def __init__(self, config: FocalDiceLossConfig | None = None):
        """
        Initialize FocalDiceLoss.

        Args:
            config: Configuration object for FocalDiceLoss.
                    If None, default values optimized for crack segmentation
                    will be used.
        """
        super().__init__()

        if config is None:
            config = FocalDiceLossConfig()

        self.config = config

        # Validate weights
        if config.focal_weight < 0 or config.dice_weight < 0:
            raise ValueError("Loss weights must be non-negative")

        if config.focal_weight + config.dice_weight <= 0:
            raise ValueError("Sum of loss weights must be positive")

        losses_config: list[dict[str, Any]] = [
            {
                "name": "focal_loss",
                "weight": config.focal_weight,
                "params": {
                    "alpha": config.focal_alpha,
                    "gamma": config.focal_gamma,
                    "reduction": config.focal_reduction,
                },
            },
            {
                "name": "dice_loss",
                "weight": config.dice_weight,
                "params": {
                    "smooth": config.dice_smooth,
                    "sigmoid": config.dice_sigmoid,
                    "eps": config.dice_eps,
                },
            },
        ]

        # Create the combined loss using the generic CombinedLoss
        self.combined_loss = CombinedLoss(
            losses_config=losses_config,
            total_loss_weight=config.total_loss_weight,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined Focal and Dice loss.

        Args:
            pred: Predicted segmentation map (logits) (B, C, H, W)
            target: Ground truth binary mask (B, C, H, W)

        Returns:
            Combined Focal and Dice loss value.
        """
        return cast(torch.Tensor, self.combined_loss(pred, target))
