# src/training/losses/dice_loss.py
import torch

from src.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss  # Import base class


@loss_registry.register(
    name="dice_loss",
    tags=["segmentation", "binary", "dice"],
)
class DiceLoss(SegmentationLoss):
    """Dice loss for binary segmentation with optional smoothing."""

    def __init__(
        self,
        smooth: float = 1.0,
        sigmoid: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor added to numerator and denominator.
                    Helps prevent division by zero if areas are 0.
            sigmoid: Whether to apply sigmoid to predictions before
                calculation.
                Set to False if predictions are already probabilities.
            eps: Small epsilon value added to denominator for numerical
                stability, especially if smooth is 0.
        """
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Dice loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (logits or probabilities)
                (B, C, H, W). Assumes C=1 for binary segmentation.
            target: Ground truth binary mask (B, C, H, W). Assumes C=1.

        Returns:
            Dice loss value (1 - Dice Coefficient)
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten predictions and targets, assuming channel is 1 (B, 1, H, W)
        # -> (B, H*W)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)

        # Dice coefficient per batch item
        dice_coeff = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth + self.eps
        )

        # Mean Dice loss over the batch
        # Clamp to ensure loss is in [0, 1]
        dice_loss = torch.clamp(1.0 - dice_coeff, min=0.0, max=1.0).mean()

        return dice_loss
