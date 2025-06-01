"""
Base combinator class with advanced features for loss combination.

This module provides a robust foundation for implementing loss combinators
with proper gradient handling, validation, and edge case management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from torch import nn

from ..interfaces.loss_interface import ILossCombinator, ILossComponent

logger = logging.getLogger(__name__)


class CombinatorError(Exception):
    """Base exception for combinator-related errors."""

    pass


class ValidationError(CombinatorError):
    """Raised when combinator validation fails."""

    pass


class NumericalStabilityError(CombinatorError):
    """Raised when numerical stability issues are detected."""

    pass


class BaseCombinator(nn.Module, ILossCombinator, ABC):
    """
    Abstract base class for loss combinators with advanced features.

    Features:
    - Automatic gradient handling and numerical stability checks
    - Comprehensive input validation
    - Support for different reduction methods
    - Edge case handling (zero weights, empty components, etc.)
    - Performance monitoring and debugging support
    - Nested combination support
    """

    def __init__(
        self,
        components: list[ILossComponent],
        validate_inputs: bool = True,
        numerical_stability_check: bool = True,
        gradient_clipping: float | None = None,
        reduction: str = "mean",
    ):
        """
        Initialize base combinator.

        Args:
            components: List of loss components to combine
            validate_inputs: Whether to validate inputs during forward pass
            numerical_stability_check: Whether to check for numerical stability
            gradient_clipping: Optional gradient clipping value
            reduction: Reduction method for final loss ('mean', 'sum', 'none')

        Raises:
            ValidationError: If components validation fails
        """
        super().__init__()  # type: ignore[reportUnknownParameterType]

        # Validate components
        self._validate_components(components)

        # Store components as ModuleList for proper parameter tracking
        self.components = nn.ModuleList(cast(list[nn.Module], components))

        # Configuration
        self.validate_inputs = validate_inputs
        self.numerical_stability_check = numerical_stability_check
        self.gradient_clipping = gradient_clipping
        self.reduction = reduction

        # Statistics for monitoring
        self._forward_count = 0
        self._numerical_warnings = 0
        self._gradient_clips = 0

        # Cache for component outputs (for debugging)
        self._last_component_outputs: list[torch.Tensor] | None = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss with validation and stability checks.

        Args:
            pred: Predicted tensor
            target: Target tensor

        Returns:
            Combined loss value

        Raises:
            ValidationError: If input validation fails
            NumericalStabilityError: If numerical stability issues detected
        """
        self._forward_count += 1

        # Input validation
        if self.validate_inputs:
            self._validate_forward_inputs(pred, target)

        # Compute component losses
        component_outputs = []
        for i, component in enumerate(self.components):
            try:
                output = component(pred, target)
                component_outputs.append(output)  # type: ignore[reportUnknownArgumentType]
            except Exception as e:
                raise CombinatorError(
                    f"Error computing loss for component {i}: {e}"
                ) from e

        # Store for debugging
        self._last_component_outputs = component_outputs

        # Numerical stability check
        if self.numerical_stability_check:
            self._check_numerical_stability(
                cast(list[torch.Tensor], component_outputs)
            )

        # Combine losses using subclass-specific logic
        combined_loss = self._combine_losses(
            cast(list[torch.Tensor], component_outputs), pred, target
        )

        # Apply reduction
        combined_loss = self._apply_reduction(combined_loss)

        # Gradient clipping if enabled
        if self.gradient_clipping is not None:
            combined_loss = self._apply_gradient_clipping(combined_loss)

        return combined_loss

    @abstractmethod
    def _combine_losses(
        self,
        component_outputs: list[torch.Tensor],
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine component losses using subclass-specific logic.

        Args:
            component_outputs: List of component loss outputs
            pred: Original predicted tensor
            target: Original target tensor

        Returns:
            Combined loss tensor
        """
        pass

    def _validate_components(self, components: list[ILossComponent]) -> None:
        """Validate component list."""
        if not components:
            raise ValidationError("Components list cannot be empty")

        for i, component in enumerate(components):
            if not callable(component):
                raise ValidationError(f"Component {i} is not callable")

            # Check if component is a PyTorch module for proper gradient
            # handling
            if not isinstance(component, nn.Module):
                logger.warning(
                    f"Component {i} is not a PyTorch module. "
                    "Gradient tracking may not work properly."
                )

    def _validate_forward_inputs(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Validate forward pass inputs."""
        # No isinstance checks necesarios, el tipador ya lo sabe

        if pred.shape != target.shape:
            raise ValidationError(
                f"Prediction shape {pred.shape} doesn't match "
                f"target shape {target.shape}"
            )

        if pred.device != target.device:
            raise ValidationError(
                f"Prediction device {pred.device} doesn't match "
                f"target device {target.device}"
            )

    def _check_numerical_stability(
        self, component_outputs: list[torch.Tensor]
    ) -> None:
        """Check for numerical stability issues."""
        for i, output in enumerate(component_outputs):
            if torch.isnan(output).any():
                raise NumericalStabilityError(
                    f"NaN detected in component {i} output"
                )

            if torch.isinf(output).any():
                raise NumericalStabilityError(
                    f"Inf detected in component {i} output"
                )

            # Check for very large values that might cause overflow
            if output.abs().max() > 1e6:
                self._numerical_warnings += 1
                logger.warning(
                    f"Large values detected in component {i} output: "
                    f"max={output.abs().max():.2e}"
                )

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to the combined loss."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def _apply_gradient_clipping(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping to prevent exploding gradients."""
        if loss.requires_grad:
            assert (
                self.gradient_clipping is not None
            ), "gradient_clipping must not be None"
            clip_value: float = self.gradient_clipping

            # Register a hook to clip gradients
            def clip_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                self._gradient_clips += 1
                return torch.clamp(grad, -clip_value, clip_value)

            loss.register_hook(clip_grad_hook)  # type: ignore[reportUnknownArgumentType]
        return loss

    def get_component_outputs(self) -> list[torch.Tensor] | None:
        """Get the last component outputs for debugging."""
        return self._last_component_outputs

    def get_statistics(self) -> dict[str, Any]:
        """Get combinator statistics for monitoring."""
        return {
            "forward_count": self._forward_count,
            "numerical_warnings": self._numerical_warnings,
            "gradient_clips": self._gradient_clips,
            "num_components": len(self.components),
            "reduction": self.reduction,
            "gradient_clipping": self.gradient_clipping,
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._forward_count = 0
        self._numerical_warnings = 0
        self._gradient_clips = 0

    def get_num_components(self) -> int:
        """Get the number of components in this combinator."""
        return len(self.components)

    def set_validation_mode(self, validate: bool) -> None:
        """Enable or disable input validation."""
        self.validate_inputs = validate

    def set_numerical_stability_check(self, check: bool) -> None:
        """Enable or disable numerical stability checking."""
        self.numerical_stability_check = check

    def __repr__(self) -> str:
        """String representation of the combinator."""
        return (
            f"{self.__class__.__name__}("
            f"components={len(self.components)}, "
            f"reduction='{self.reduction}', "
            f"gradient_clipping={self.gradient_clipping})"
        )


class CombinatorFactory:
    """
    Factory for creating loss combinators with common configurations.
    """

    @staticmethod
    def create_weighted_sum(
        components: list[ILossComponent],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> "BaseCombinator":
        from .enhanced_weighted_sum import EnhancedWeightedSumCombinator

        return EnhancedWeightedSumCombinator(components, weights, **kwargs)

    @staticmethod
    def create_product(
        components: list[ILossComponent], **kwargs: Any
    ) -> "BaseCombinator":
        from .enhanced_product import EnhancedProductCombinator

        return EnhancedProductCombinator(components, **kwargs)

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> "BaseCombinator":
        """Create combinator from configuration dictionary."""
        combinator_type = config.get("type")
        components = config.get("components", [])

        if combinator_type == "sum":
            weights = config.get("weights")
            return CombinatorFactory.create_weighted_sum(components, weights)
        elif combinator_type == "product":
            return CombinatorFactory.create_product(components)
        else:
            raise ValueError(f"Unknown combinator type: {combinator_type}")


# Utility functions for edge case handling
def handle_zero_weights(weights: list[float]) -> list[float]:
    """Handle zero weights by replacing with small epsilon values."""
    epsilon = 1e-8
    processed_weights: list[float] = []

    for weight in weights:
        if abs(weight) < epsilon:
            logger.warning(
                f"Zero weight detected, replacing with epsilon={epsilon}"
            )
            processed_weights.append(epsilon)
        else:
            processed_weights.append(weight)

    return processed_weights


def normalize_weights(weights: list[float]) -> list[float]:
    """Normalize weights to sum to 1.0 with numerical stability."""
    weights = handle_zero_weights(weights)
    total = sum(weights)

    if total <= 0:
        raise ValueError("Sum of weights must be positive")

    return [w / total for w in weights]


def validate_component_compatibility(
    components: list[ILossComponent], pred: torch.Tensor, target: torch.Tensor
) -> bool:
    """
    Validate that all components are compatible with given inputs.

    Args:
        components: List of loss components
        pred: Prediction tensor
        target: Target tensor

    Returns:
        True if all components are compatible

    Raises:
        ValidationError: If any component is incompatible
    """
    for i, component in enumerate(components):
        try:
            # Test component with small sample
            test_pred = pred[:1] if pred.dim() > 0 else pred
            test_target = target[:1] if target.dim() > 0 else target
            _ = component(test_pred, test_target)
        except Exception as e:
            raise ValidationError(
                f"Component {i} is incompatible with input shapes: {e}"
            ) from e

    return True
