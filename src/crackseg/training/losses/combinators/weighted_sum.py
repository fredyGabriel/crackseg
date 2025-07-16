"""
Weighted sum combinator for loss functions.
"""

from typing import cast

import torch
from torch import nn

from ..interfaces.loss_interface import ILossCombinator, ILossComponent


class WeightedSumCombinator(nn.Module, ILossCombinator):
    """
    Combines multiple loss components using weighted sum.

    This combinator computes a weighted sum of loss values from multiple
    loss components, with automatic weight normalization.
    """

    def __init__(
        self,
        components: list[ILossComponent],
        weights: list[float] | None = None,
    ):
        """
        Initialize weighted sum combinator.

        Args:
            components: List of loss components to combine
            weights: Optional weights for each component. If None, equal
                weights are used. Weights are automatically normalized to
                sum to 1.0.

        Raises:
            ValueError: If components is empty or weights length doesn't match
            components
        """
        super().__init__()

        if not components:
            raise ValueError("Components list cannot be empty")

        # Store components as ModuleList for proper parameter tracking
        self.components = nn.ModuleList(cast(list[nn.Module], components))

        # Handle weights
        if weights is None:
            self.weights = [1.0 / len(components)] * len(components)
        else:
            if len(weights) != len(components):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of components ({len(components)})"
                )

            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("Sum of weights must be positive")

            # Normalize weights
            self.weights = [w / weight_sum for w in weights]

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted sum of component losses.

        Args:
            pred: Predicted tensor
            target: Target tensor

        Returns:
            Weighted sum of component loss values
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for weight, component in zip(
            self.weights, self.components, strict=True
        ):
            component_loss = component(pred, target)
            total_loss += weight * component_loss

        return total_loss

    def get_component_weights(self) -> list[float]:
        """Get the normalized weights for each component."""
        return self.weights.copy()

    def get_num_components(self) -> int:
        """Get the number of components in this combinator."""
        return len(self.components)
