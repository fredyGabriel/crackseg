"""
Enhanced weighted sum combinator with advanced features.

This module provides a robust weighted sum combinator with numerical stability,
edge case handling, and comprehensive validation.
"""

import logging
from typing import Any

import torch

from ..interfaces.loss_interface import ILossComponent
from .base_combinator import (
    BaseCombinator,
    handle_zero_weights,
    normalize_weights,
)

logger = logging.getLogger(__name__)


class EnhancedWeightedSumCombinator(BaseCombinator):
    """
    Enhanced weighted sum combinator with advanced features.

    Features:
    - Automatic weight normalization with numerical stability
    - Zero weight handling with epsilon replacement
    - Dynamic weight adjustment during training
    - Support for different reduction methods
    - Comprehensive validation and error handling
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        components: list[ILossComponent],
        weights: list[float] | None = None,
        auto_normalize: bool = True,
        handle_zero_weights_flag: bool = True,
        weight_epsilon: float = 1e-8,
        **kwargs: Any,
    ):
        """
        Initialize enhanced weighted sum combinator.

        Args:
            components: List of loss components to combine
            weights: Optional weights for each component. If None, equal
                    weights are used.
            auto_normalize: Whether to automatically normalize weights to sum
                    to 1.0
            handle_zero_weights_flag: Whether to handle zero weights with
                    epsilon
            weight_epsilon: Epsilon value for zero weight replacement
            **kwargs: Additional arguments passed to BaseCombinator

        Raises:
            ValueError: If weights configuration is invalid
        """
        super().__init__(components, **kwargs)

        self.auto_normalize = auto_normalize
        self.handle_zero_weights_flag = handle_zero_weights_flag
        self.weight_epsilon = weight_epsilon

        # Initialize weights
        self.weights = self._initialize_weights(weights)

        # Store original weights for reference
        self._original_weights = weights.copy() if weights else None

    def _initialize_weights(self, weights: list[float] | None) -> list[float]:
        """Initialize and process weights."""
        num_components = len(self.components)

        if weights is None:
            # Equal weights
            weights = [1.0 / num_components] * num_components
        else:
            if len(weights) != num_components:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of components ({num_components})"
                )

            weights = list(weights)  # Make a copy

        # Handle zero weights if enabled
        if self.handle_zero_weights_flag:
            weights = handle_zero_weights(weights)

        # Normalize weights if enabled
        if self.auto_normalize:
            weights = normalize_weights(weights)

        return weights

    def _combine_losses(
        self,
        component_outputs: list[torch.Tensor],
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine component losses using weighted sum.

        Args:
            component_outputs: List of component loss outputs
            pred: Original predicted tensor (unused in weighted sum)
            target: Original target tensor (unused in weighted sum)

        Returns:
            Weighted sum of component losses
        """
        # Initialize with proper device and dtype
        device = component_outputs[0].device
        dtype = component_outputs[0].dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # Compute weighted sum
        for weight, component_output in zip(
            self.weights, component_outputs, strict=True
        ):
            weighted_loss = weight * component_output
            total_loss = total_loss + weighted_loss

        return total_loss

    def get_component_weights(self) -> list[float]:
        """Get the current normalized weights for each component."""
        return self.weights.copy()

    def get_original_weights(self) -> list[float] | None:
        """Get the original weights before processing."""
        return (
            self._original_weights.copy() if self._original_weights else None
        )

    def update_weights(self, new_weights: list[float]) -> None:
        """
        Update weights with validation and processing.

        Args:
            new_weights: New weights to set

        Raises:
            ValueError: If new weights are invalid
        """
        if len(new_weights) != len(self.components):
            raise ValueError(
                f"Number of new weights ({len(new_weights)}) must match "
                f"number of components ({len(self.components)})"
            )

        # Process new weights
        processed_weights = list(new_weights)

        if self.handle_zero_weights_flag:
            processed_weights = handle_zero_weights(processed_weights)

        if self.auto_normalize:
            processed_weights = normalize_weights(processed_weights)

        self.weights = processed_weights
        logger.info(f"Updated weights: {self.weights}")

    def get_weight_statistics(self) -> dict[str, float]:
        """Get statistics about the current weights."""
        weights_tensor = torch.tensor(self.weights)
        return {
            "mean": weights_tensor.mean().item(),
            "std": weights_tensor.std().item(),
            "min": weights_tensor.min().item(),
            "max": weights_tensor.max().item(),
            "sum": weights_tensor.sum().item(),
            "entropy": self._compute_weight_entropy(),
        }

    def _compute_weight_entropy(self) -> float:
        """Compute entropy of weight distribution."""
        weights_tensor = torch.tensor(self.weights)
        # Add small epsilon to avoid log(0)
        weights_tensor = weights_tensor + 1e-10
        entropy = -(weights_tensor * torch.log(weights_tensor)).sum().item()
        return entropy

    def balance_weights(self) -> None:
        """Reset weights to equal distribution."""
        num_components = len(self.components)
        self.weights = [1.0 / num_components] * num_components
        logger.info("Weights balanced to equal distribution")

    def set_dominant_component(
        self, component_index: int, dominance: float = 0.8
    ) -> None:
        """
        Set one component to be dominant with specified weight.

        Args:
            component_index: Index of component to make dominant
            dominance: Weight for the dominant component (0.0 to 1.0)

        Raises:
            ValueError: If component_index is invalid or dominance is out of
            range
        """
        if not 0 <= component_index < len(self.components):
            raise ValueError(f"Component index {component_index} out of range")

        if not 0.0 <= dominance <= 1.0:
            raise ValueError(
                f"Dominance {dominance} must be between 0.0 and 1.0"
            )

        # Calculate remaining weight for other components
        remaining_weight = 1.0 - dominance
        other_weight = remaining_weight / (len(self.components) - 1)

        # Set new weights
        new_weights = [other_weight] * len(self.components)
        new_weights[component_index] = dominance

        self.update_weights(new_weights)
        logger.info(
            f"Set component {component_index} as dominant with weight "
            f"{dominance}"
        )

    def __repr__(self) -> str:
        """String representation of the enhanced weighted sum combinator."""
        return (
            f"EnhancedWeightedSumCombinator("
            f"components={len(self.components)}, "
            f"weights={[f'{w:.3f}' for w in self.weights]}, "
            f"auto_normalize={self.auto_normalize}, "
            f"reduction='{self.reduction}')"
        )


# Convenience function for creating enhanced weighted sum combinators
def create_weighted_sum_combinator(
    components: list[ILossComponent],
    weights: list[float] | None = None,
    **kwargs: Any,
) -> EnhancedWeightedSumCombinator:
    """
    Convenience function to create an enhanced weighted sum combinator.

    Args:
        components: List of loss components to combine
        weights: Optional weights for each component
        **kwargs: Additional arguments passed to the combinator

    Returns:
        EnhancedWeightedSumCombinator instance
    """
    return EnhancedWeightedSumCombinator(components, weights, **kwargs)
