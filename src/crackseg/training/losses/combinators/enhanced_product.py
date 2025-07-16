"""
Enhanced product combinator with advanced features.

This module provides a robust product combinator with numerical stability,
edge case handling, and comprehensive validation.
"""

import logging
from typing import Any, cast

import torch

from ..interfaces.loss_interface import ILossComponent
from .base_combinator import BaseCombinator

logger = logging.getLogger(__name__)


class EnhancedProductCombinator(BaseCombinator):
    """
    Enhanced product combinator with advanced features.

    Features:
    - Numerical stability for product operations
    - Log-space computation to prevent overflow/underflow
    - Zero loss handling with epsilon replacement
    - Support for different reduction methods
    - Comprehensive validation and error handling
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        components: list[ILossComponent],
        use_log_space: bool = True,
        zero_epsilon: float = 1e-8,
        stability_threshold: float = 1e-6,
        **kwargs: Any,
    ):
        """
        Initialize enhanced product combinator.

        Args:
            components: List of loss components to combine
            use_log_space: Whether to use log-space computation for numerical
                            stability
            zero_epsilon: Epsilon value for zero loss replacement
            stability_threshold: Threshold for numerical stability warnings
            **kwargs: Additional arguments passed to BaseCombinator

        Raises:
            ValueError: If components configuration is invalid
        """
        super().__init__(components, **kwargs)

        self.use_log_space = use_log_space
        self.zero_epsilon = zero_epsilon
        self.stability_threshold = stability_threshold

        # Statistics for monitoring
        self._zero_replacements = 0
        self._stability_warnings = 0

    def _combine_losses(
        self,
        component_outputs: list[torch.Tensor],
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine component losses using product operation.

        Args:
            component_outputs: List of component loss outputs
            pred: Original predicted tensor (unused in product)
            target: Original target tensor (unused in product)

        Returns:
            Product of component losses
        """
        if self.use_log_space:
            return self._compute_log_space_product(component_outputs)
        else:
            return self._compute_direct_product(component_outputs)

    def _compute_log_space_product(
        self, component_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute product in log space for numerical stability.

        Args:
            component_outputs: List of component loss outputs

        Returns:
            Product computed in log space
        """
        # Handle zero and negative values
        processed_outputs: list[torch.Tensor] = []
        for i, output in enumerate(component_outputs):
            # Replace zeros and very small values with epsilon
            output_processed = torch.where(
                output < self.zero_epsilon,
                torch.tensor(
                    self.zero_epsilon, device=output.device, dtype=output.dtype
                ),
                output,
            )

            # Count zero replacements
            if (output < self.zero_epsilon).any():
                self._zero_replacements += 1
                logger.debug(
                    f"Replaced {(output < self.zero_epsilon).sum()} zero "
                    f"values in component {i}"
                )

            processed_outputs.append(output_processed)

        # Compute log of each component
        log_outputs = [torch.log(output) for output in processed_outputs]

        # Sum logs (equivalent to product in linear space)
        log_sum = torch.stack(log_outputs).sum(dim=0)

        # Convert back to linear space
        result = torch.exp(log_sum)

        # Check for numerical issues
        if torch.isnan(result).any() or torch.isinf(result).any():
            self._stability_warnings += 1
            logger.warning(
                "Numerical instability detected in log-space product "
                "computation"
            )

        return result

    def _compute_direct_product(
        self, component_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute product directly (may be less numerically stable).

        Args:
            component_outputs: List of component loss outputs

        Returns:
            Direct product of component losses
        """
        # Initialize with the first component
        result = component_outputs[0]

        # Handle zero values in first component
        if (result < self.zero_epsilon).any():
            self._zero_replacements += 1
            result = torch.where(
                result < self.zero_epsilon,
                torch.tensor(
                    self.zero_epsilon, device=result.device, dtype=result.dtype
                ),
                result,
            )

        # Multiply with remaining components
        for i, component_output in enumerate(component_outputs[1:], 1):
            # Handle zero values
            processed_output = torch.where(
                component_output < self.zero_epsilon,
                torch.tensor(
                    self.zero_epsilon,
                    device=component_output.device,
                    dtype=component_output.dtype,
                ),
                component_output,
            )

            if (component_output < self.zero_epsilon).any():
                self._zero_replacements += 1
                logger.debug(f"Replaced zero values in component {i}")

            result = result * processed_output

            # Check for numerical issues after each multiplication
            if torch.isnan(result).any() or torch.isinf(result).any():
                self._stability_warnings += 1
                logger.warning(
                    f"Numerical instability detected after multiplying "
                    f"component {i}"
                )

        return result

    def get_product_statistics(self) -> dict[str, Any]:
        """Get statistics about the product computation."""
        return {
            "zero_replacements": self._zero_replacements,
            "stability_warnings": self._stability_warnings,
            "use_log_space": self.use_log_space,
            "zero_epsilon": self.zero_epsilon,
            "stability_threshold": self.stability_threshold,
        }

    def reset_product_statistics(self) -> None:
        """Reset product-specific statistics."""
        self._zero_replacements = 0
        self._stability_warnings = 0

    def set_log_space_mode(self, use_log_space: bool) -> None:
        """
        Enable or disable log-space computation.

        Args:
            use_log_space: Whether to use log-space computation
        """
        self.use_log_space = use_log_space
        logger.info(
            f"Log-space computation {
                'enabled' if use_log_space else 'disabled'
            }"
        )

    def set_zero_epsilon(self, epsilon: float) -> None:
        """
        Set the epsilon value for zero replacement.

        Args:
            epsilon: New epsilon value

        Raises:
            ValueError: If epsilon is not positive
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")

        self.zero_epsilon = epsilon
        logger.info(f"Zero epsilon set to {epsilon}")

    def analyze_component_ranges(
        self, component_outputs: list[torch.Tensor]
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze the ranges of component outputs for debugging.

        Args:
            component_outputs: List of component loss outputs

        Returns:
            Dictionary with range analysis for each component
        """
        analysis = {}
        for i, output in enumerate(component_outputs):
            analysis[f"component_{i}"] = {
                "min": output.min().item(),
                "max": output.max().item(),
                "mean": output.mean().item(),
                "std": output.std().item(),
                "zeros": (output == 0).sum().item(),
                "near_zeros": (output < self.zero_epsilon).sum().item(),
                "shape": list(output.shape),
            }

        return cast(dict[str, dict[str, Any]], analysis)

    def get_geometric_mean_equivalent(
        self, component_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute geometric mean equivalent of the product.

        This can be useful for understanding the "average" contribution
        of each component to the final product.

        Args:
            component_outputs: List of component loss outputs

        Returns:
            Geometric mean of component losses
        """
        n = len(component_outputs)
        if self.use_log_space:
            # Geometric mean in log space
            log_outputs = [
                torch.log(torch.clamp(output, min=self.zero_epsilon))
                for output in component_outputs
            ]
            log_mean = torch.stack(log_outputs).mean(dim=0)
            return torch.exp(log_mean)
        else:
            # Direct geometric mean computation
            product = self._compute_direct_product(component_outputs)
            return torch.pow(product, 1.0 / n)

    def __repr__(self) -> str:
        """String representation of the enhanced product combinator."""
        return (
            f"EnhancedProductCombinator("
            f"components={len(self.components)}, "
            f"use_log_space={self.use_log_space}, "
            f"zero_epsilon={self.zero_epsilon}, "
            f"reduction='{self.reduction}')"
        )


# Convenience function for creating enhanced product combinators
def create_product_combinator(
    components: list[ILossComponent], **kwargs: Any
) -> EnhancedProductCombinator:
    """
    Convenience function to create an enhanced product combinator.

    Args:
        components: List of loss components to combine
        **kwargs: Additional arguments passed to the combinator

    Returns:
        EnhancedProductCombinator instance
    """
    return EnhancedProductCombinator(components, **kwargs)
