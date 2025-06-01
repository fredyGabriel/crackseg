# src/training/losses/combined_loss.py
from typing import Any

import torch
from torch import nn

from src.training.losses.loss_registry_setup import loss_registry

from .base_loss import SegmentationLoss  # Import base class


@loss_registry.register(
    name="combined_loss",
    tags=["segmentation", "utility", "meta"],
)
class CombinedLoss(SegmentationLoss):
    """
    Combined loss function that applies multiple loss functions with weights.
    Losses are expected to be nn.Module instances.
    """

    def __init__(
        self,
        losses_config: list[dict[str, Any]],
        total_loss_weight: float = 1.0,
    ):
        """
        Initialize combined loss function.

        Args:
            losses_config: A list of dictionaries, where each dictionary
                        defines a loss component.
                        Each dict must have a 'name' key (registered name of
                        the loss) and a 'weight' key. It can optionally have a
                        'params' key (dict) for loss constructor arguments.
                           Example: [
                               {'name': 'dice_loss', 'weight': 0.5,
                               'params': {'smooth': 1.0}},
                               {'name': 'focal_loss', 'weight': 0.5,
                               'params': {'alpha': 0.25, 'gamma': 2.0}}
                           ]
            total_loss_weight: A global weight for the final combined loss.
            Default is 1.0.
        """
        super().__init__()
        self.loss_modules = nn.ModuleList()
        # Store weights separately for clarity, though they are fixed after
        # init
        self.weights: list[float] = []
        self.total_loss_weight = total_loss_weight

        if not losses_config:
            raise ValueError("losses_config cannot be empty for CombinedLoss.")

        parsed_weights = []
        for config in losses_config:
            if "name" not in config or "weight" not in config:
                raise ValueError(
                    "Each item in losses_config must be a dict "
                    "with 'name' and 'weight' keys."
                )

            name = config["name"]
            weight = config["weight"]
            params = config.get("params", {})

            try:
                # Instantiate loss from the global registry
                loss_module = loss_registry.instantiate(name, **params)
            except KeyError as err:
                raise ValueError(
                    f"Loss '{name}' not found in registry. Ensure it's "
                    "registered."
                ) from err
            except Exception as e:
                raise ValueError(
                    f"Error instantiating loss '{name}' with params {params}: "
                    f"{e}"
                ) from e

            self.loss_modules.append(loss_module)
            parsed_weights.append(weight)

        # Normalize weights if they don't sum to 1.0, for internal consistency
        # of the component weights
        sum_of_weights = sum(parsed_weights)
        if sum_of_weights <= 0:
            raise ValueError(
                "Sum of weights in losses_config must be positive."
            )

        self.weights = [w / sum_of_weights for w in parsed_weights]

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss between predicted and target masks.

        Args:
            pred: Predicted segmentation map (logits or probabilities)
            (B, C, H, W)
            target: Ground truth binary mask (B, C, H, W)

        Returns:
            Combined loss value, scaled by total_loss_weight.
        """
        final_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for i, loss_fn in enumerate(self.loss_modules):
            component_loss = loss_fn(pred, target)
            final_loss += self.weights[i] * component_loss

        return final_loss * self.total_loss_weight
