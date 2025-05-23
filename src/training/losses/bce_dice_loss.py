# src/training/losses/bce_dice_loss.py
from dataclasses import dataclass  # Added dataclass and field
from typing import Any, cast

import torch

from src.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss  # Import base class
from .combined_loss import (
    CombinedLoss,  # To use CombinedLoss as its implementation detail
)


@dataclass
class BCEDiceLossConfig:
    """Configuration for BCEDiceLoss."""

    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_smooth: float = 1.0
    dice_sigmoid: bool = True
    dice_eps: float = 1e-6
    bce_reduction: str = "mean"
    bce_pos_weight: torch.Tensor | None = None


@loss_registry.register(
    name="bce_dice_loss",
    tags=["segmentation", "binary", "hybrid", "bce", "dice"],
)
class BCEDiceLoss(SegmentationLoss):
    """
    A combined loss function that is a specific instance of CombinedLoss,
    summing BCE and Dice losses. This version assumes logits as input for BCE.
    """

    def __init__(self, config: BCEDiceLossConfig | None = None):
        """
        Initialize BCEDiceLoss.

        Args:
            config: Configuration object for BCEDiceLoss.
                    If None, default values will be used.
        """
        super().__init__()

        if config is None:
            config = BCEDiceLossConfig()

        self.config = config  # Store config if needed later

        losses_config: list[dict[str, Any]] = [
            {
                "name": "bce_loss",
                "weight": config.bce_weight,
                "params": {
                    "reduction": config.bce_reduction,
                    "weight": config.bce_pos_weight,
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

        # The CombinedLoss will normalize these weights internally if they
        # don't sum to 1, but it is clearer if they are provided as such.
        # The final loss is a weighted sum.
        self.combined_loss = CombinedLoss(
            losses_config=losses_config, total_loss_weight=1.0
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined BCE and Dice loss.

        Args:
            pred: Predicted segmentation map (logits) (B, C, H, W)
            target: Ground truth binary mask (B, C, H, W)

        Returns:
            Combined BCE and Dice loss value.
        """
        return cast(torch.Tensor, self.combined_loss(pred, target))
