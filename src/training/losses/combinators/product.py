"""
Product combinator for loss functions.
"""

from typing import cast

import torch
from torch import nn

from ..interfaces.loss_interface import ILossCombinator, ILossComponent


class ProductCombinator(nn.Module, ILossCombinator):
    """
    Combines multiple loss components using element-wise product.

    This combinator computes the product of loss values from multiple
    loss components. Useful for cases where all losses must be minimized
    simultaneously.
    """

    def __init__(self, components: list[ILossComponent]):
        """
        Initialize product combinator.

        Args:
            components: List of loss components to combine

        Raises:
            ValueError: If components is empty
        """
        super().__init__()

        if not components:
            raise ValueError("Components list cannot be empty")

        # Store components as ModuleList for proper parameter tracking
        self.components = nn.ModuleList(cast(list[nn.Module], components))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute product of component losses.

        Args:
            pred: Predicted tensor
            target: Target tensor

        Returns:
            Product of component loss values
        """
        # Initialize with the first component
        result = self.components[0](pred, target)

        # Multiply with remaining components
        for component in self.components[1:]:
            component_loss = component(pred, target)
            result = result * component_loss

        return cast(torch.Tensor, result)

    def get_num_components(self) -> int:
        """Get the number of components in this combinator."""
        return len(self.components)
