import torch
import torch.optim

from crackseg.training.optimizers.registry import register_optimizer


@register_optimizer("custom_adam")
class CustomAdam(torch.optim.Adam):
    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.custom_parameter = 0.1  # Custom parameter

    def step(self, closure=None):
        # Add custom logic here if needed
        return super().step(closure)
