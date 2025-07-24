import torch
import torch.nn as nn

from crackseg.training.losses.base_loss import SegmentationLoss
from crackseg.training.losses.loss_registry_setup import loss_registry


@loss_registry.register(
    name="smooth_l1_loss",
    tags=["regression", "smooth_l1", "huber"],
    force=True,
)
class SmoothL1Loss(SegmentationLoss):
    def __init__(self, beta: float = 1.0, config=None, **kwargs):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=self.beta)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(y_pred, y_true)
