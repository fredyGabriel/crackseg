# src/training/losses/focal_loss.py
import torch
import torch.nn.functional as F  # For F.binary_cross_entropy_with_logits

from src.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss  # Import base class


@loss_registry.register(
    name="focal_loss",
    tags=["segmentation", "binary", "focal", "imbalance"],
)
class FocalLoss(SegmentationLoss):
    """
    Focal Loss for dealing with class imbalance in segmentation tasks.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the ground truth class.
    Assumes input `pred` are logits.
    """

    def __init__(
        self,
        alpha: float = 0.25,  # Alpha balances positive/negative examples
        gamma: float = 2.0,  # Gamma focuses on hard examples
        reduction: str = "mean",
        eps: float = 1e-6,  # For num. stability if not using BCEWithLogits
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for the positive class. Default: 0.25.
                   If None, no alpha weighting is applied.
            gamma: Focusing parameter to down-weight easy examples. Default:
                    2.0.
            reduction: Specifies the reduction to apply to the output:
                       'none' | 'mean' | 'sum'. Default: 'mean'.
            eps: Epsilon for numerical stability if sigmoid is applied
                manually. Not typically needed if using BCEWithLogitsLoss.
        """
        super().__init__()
        if alpha is not None and not (0 < alpha < 1):
            raise ValueError("Alpha should be in (0,1) or None to disable it.")
        if gamma < 0:
            raise ValueError("Gamma should be non-negative.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Kept for potential direct sigmoid use, though BCEWithLogits is
        # preferred
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Focal Loss between predicted logits and target masks.

        Args:
            pred: Predicted segmentation map (logits) (B, C, H, W).
                  Assumes C=1 for binary segmentation.
            target: Ground truth binary mask (B, C, H, W). Assumes C=1.

        Returns:
            Focal loss value based on the specified reduction.
        """
        # Ensure target is float for BCEWithLogitsLoss
        target = target.float()

        # Calculate BCE loss without reduction, using logits directly for
        # stability
        # This is equivalent to: -log(sigmoid(x)) for target 1,
        # -log(1-sigmoid(x)) for target 0
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )

        # Get probabilities p_t (probability of the true class)
        # pred_sigmoid = torch.sigmoid(pred)
        # p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        # A more direct way since pred are logits and target is 0 or 1:
        # if target == 1, p_t = sigmoid(pred). If target == 0,
        # p_t = 1 - sigmoid(pred)
        # This is equivalent to exp(-bce_loss) because bce_loss = -log(p_t)
        p_t = torch.exp(-bce_loss)

        # Calculate focal loss term: (1 - p_t)^gamma
        focal_term = (1 - p_t).pow(self.gamma)
        loss = focal_term * bce_loss

        if self.alpha is not None:
            # Create alpha_t: alpha for true class, (1-alpha) for other class
            alpha_t = torch.full_like(target, self.alpha)
            alpha_t = torch.where(target == 1, alpha_t, 1.0 - alpha_t)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")
