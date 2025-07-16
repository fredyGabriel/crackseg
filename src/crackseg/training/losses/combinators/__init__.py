"""
Loss combinators for composing multiple loss functions.
This module provides implementations for combining multiple loss components
in various ways (weighted sum, product, etc.).
"""

from .base_combinator import (
    BaseCombinator,
    CombinatorError,
    CombinatorFactory,
    NumericalStabilityError,
    ValidationError,
)
from .enhanced_product import (
    EnhancedProductCombinator,
    create_product_combinator,
)
from .enhanced_weighted_sum import (
    EnhancedWeightedSumCombinator,
    create_weighted_sum_combinator,
)
from .product import ProductCombinator
from .weighted_sum import WeightedSumCombinator

__all__ = [
    # Base classes and factory
    "BaseCombinator",
    "CombinatorFactory",
    "CombinatorError",
    "ValidationError",
    "NumericalStabilityError",
    # Enhanced combinators (recommended)
    "EnhancedWeightedSumCombinator",
    "EnhancedProductCombinator",
    "create_weighted_sum_combinator",
    "create_product_combinator",
    # Legacy combinators (for backward compatibility)
    "WeightedSumCombinator",
    "ProductCombinator",
]
