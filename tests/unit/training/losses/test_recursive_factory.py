"""
Legacy test file - SPLIT INTO SMALLER MODULES. This file originally
contained 802 lines with comprehensive tests for the recursive loss
factory. It has been refactored into focused test modules: 1.
test_recursive_factory_basic.py - Basic instantiation tests (~130
lines) 2. test_recursive_factory_combinations.py - Combination tests
(~250 lines) 3. test_recursive_factory_errors.py - Error handling
tests (~130 lines) 4. test_recursive_factory_config.py - Configuration
validation tests (~120 lines) 5. test_recursive_factory_performance.py
- Performance and gradient tests (~200 lines) 6.
test_recursive_factory_regression.py - Regression scenario tests (~170
lines) Total: 6 focused files instead of 1 large monolith. This
refactoring improves: - Code maintainability and navigation - Test
organization and readability - Parallel test execution capability -
Focused test failure analysis Run all recursive factory tests with:
pytest tests/unit/training/losses/test_recursive_factory_*.py
"""

# Import all test modules to ensure they are discovered by pytest
# This maintains backward compatibility for test runners expecting this file
import pytest

# Re-export all test classes for compatibility
try:
    from tests.unit.training.losses.test_recursive_factory_basic import (
        TestBasicLossInstantiation,
    )
    from tests.unit.training.losses.test_recursive_factory_combinations import (  # noqa: E501
        TestNestedCombinations,
        TestSimpleCombinations,
    )
    from tests.unit.training.losses.test_recursive_factory_config import (
        TestConfigurationValidation,
    )
    from tests.unit.training.losses.test_recursive_factory_errors import (
        TestEdgeCasesAndErrorHandling,
    )
    from tests.unit.training.losses.test_recursive_factory_performance import (
        TestGradientFlow,
        TestPerformanceAndMemory,
    )
    from tests.unit.training.losses.test_recursive_factory_regression import (
        TestRegressionTests,
    )

    # Mark all classes as available for backward compatibility
    __all__ = [
        "TestBasicLossInstantiation",
        "TestSimpleCombinations",
        "TestNestedCombinations",
        "TestEdgeCasesAndErrorHandling",
        "TestConfigurationValidation",
        "TestPerformanceAndMemory",
        "TestGradientFlow",
        "TestRegressionTests",
    ]

except ImportError as e:
    # If split files are not available, provide helpful error message
    pytest.fail(
        f"Recursive factory test modules not found: {e}\n"
        "This file has been split into smaller modules. "
        "Ensure all test_recursive_factory_*.py files are present."
    )
