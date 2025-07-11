# This file makes src.training.losses a Python package
# and ensures all defined loss modules are imported and thus registered.

from .base_loss import SegmentationLoss
from .bce_dice_loss import BCEDiceLoss
from .bce_loss import BCELoss
from .combined_loss import CombinedLoss
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss

__all__ = [
    "SegmentationLoss",
    "BCELoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "BCEDiceLoss",
]
