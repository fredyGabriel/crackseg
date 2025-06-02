#!/usr/bin/env python3
"""
Integration tests for the new clean loss factory architecture.
Tests the complete system working together without circular dependencies.
"""

from typing import Any

import pytest
import torch


@pytest.fixture
def clean_registry() -> Any:
    """Create a clean registry with test losses for isolated testing."""
    # Import directly to avoid circular dependencies
    # Add explicit path manipulation to avoid src.__init__.py
    import importlib.util
    import os

    # Get the absolute path to the registry module
    registry_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "registry",
        "clean_registry.py",
    )
    registry_path = os.path.abspath(registry_path)

    spec = importlib.util.spec_from_file_location(
        "clean_registry", registry_path
    )
    if spec is None:
        raise RuntimeError(f"Could not load spec for {registry_path}")
    registry_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(registry_module)

    CleanLossRegistry = registry_module.CleanLossRegistry
    registry = CleanLossRegistry()

    # Register a simple mock loss for testing
    def mock_dice_loss(**params: Any) -> torch.nn.Module:
        class MockDiceLoss(torch.nn.Module):
            def __init__(self, smooth: float = 1.0):
                super().__init__()
                self.smooth = smooth

            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.5, requires_grad=True)

        return MockDiceLoss(**params)

    def mock_bce_loss(**params: Any) -> torch.nn.Module:
        class MockBCELoss(torch.nn.Module):
            def __init__(self, reduction: str = "mean"):
                super().__init__()
                self.reduction = reduction

            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.3, requires_grad=True)

        return MockBCELoss(**params)

    registry.register_factory("dice_loss", mock_dice_loss, tags=["test"])
    registry.register_factory("bce_loss", mock_bce_loss, tags=["test"])

    return registry


@pytest.fixture
def sample_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample prediction and target tensors for testing."""
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    return pred, target


def test_clean_registry_functionality(clean_registry: Any) -> None:
    """Test the clean registry basic functionality."""
    assert clean_registry.is_registered("dice_loss")
    assert clean_registry.is_registered("bce_loss")
    assert "dice_loss" in clean_registry.list_available()

    # Test instantiation
    loss_fn = clean_registry.instantiate("dice_loss", smooth=1.0)
    assert loss_fn is not None
    assert hasattr(loss_fn, "smooth")
    assert loss_fn.smooth == 1.0


def test_weighted_sum_combinator(
    clean_registry: Any, sample_tensors: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Test WeightedSumCombinator integration."""
    # Direct import using importlib to avoid circular dependencies
    import importlib.util
    import os

    combinator_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "combinators",
        "weighted_sum.py",
    )
    combinator_path = os.path.abspath(combinator_path)

    spec = importlib.util.spec_from_file_location(
        "weighted_sum", combinator_path
    )
    if spec is None:
        raise RuntimeError(f"Could not load spec for {combinator_path}")
    combinator_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(combinator_module)

    WeightedSumCombinator = combinator_module.WeightedSumCombinator

    # Create individual loss components
    dice_loss = clean_registry.instantiate("dice_loss", smooth=1.0)
    bce_loss = clean_registry.instantiate("bce_loss", reduction="mean")

    # Create combinator
    combinator = WeightedSumCombinator(
        [dice_loss, bce_loss], weights=[0.6, 0.4]
    )

    # Test forward pass
    pred, target = sample_tensors
    result = combinator(pred, target)

    assert isinstance(result, torch.Tensor)
    assert result.requires_grad

    # Check that weights are normalized
    weights = combinator.get_component_weights()
    assert abs(sum(weights) - 1.0) < 1e-6


def test_product_combinator(
    clean_registry: Any, sample_tensors: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Test ProductCombinator integration."""
    # Direct import using importlib to avoid circular dependencies
    import importlib.util
    import os

    combinator_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "combinators",
        "product.py",
    )
    combinator_path = os.path.abspath(combinator_path)

    spec = importlib.util.spec_from_file_location("product", combinator_path)
    if spec is None:
        raise RuntimeError(f"Could not load spec for {combinator_path}")
    combinator_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(combinator_module)

    ProductCombinator = combinator_module.ProductCombinator

    # Create individual loss components
    dice_loss = clean_registry.instantiate("dice_loss", smooth=1.0)
    bce_loss = clean_registry.instantiate("bce_loss", reduction="mean")

    # Create combinator
    combinator = ProductCombinator([dice_loss, bce_loss])

    # Test forward pass
    pred, target = sample_tensors
    result = combinator(pred, target)

    assert isinstance(result, torch.Tensor)
    assert result.requires_grad
    assert combinator.get_num_components() == 2


def test_recursive_factory_integration(
    clean_registry: Any, sample_tensors: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Test the complete recursive factory integration."""
    # Direct imports using importlib to avoid circular dependencies
    import importlib.util
    import os

    # Import ConfigValidator
    validator_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "factory",
        "config_validator.py",
    )
    validator_path = os.path.abspath(validator_path)

    spec = importlib.util.spec_from_file_location(
        "config_validator", validator_path
    )
    if spec is None:
        raise RuntimeError(f"Could not load spec for {validator_path}")
    validator_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(validator_module)

    ConfigValidator = validator_module.ConfigValidator

    # Import RecursiveLossFactory
    factory_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "factory",
        "recursive_factory.py",
    )
    factory_path = os.path.abspath(factory_path)

    spec = importlib.util.spec_from_file_location(
        "recursive_factory", factory_path
    )
    if spec is None:
        raise RuntimeError(f"Could not load spec for {factory_path}")
    factory_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(factory_module)

    RecursiveLossFactory = factory_module.RecursiveLossFactory

    # Create factory with our test registry
    factory = RecursiveLossFactory()
    factory.registry = clean_registry
    factory.validator = ConfigValidator(clean_registry)

    # Test simple leaf loss
    leaf_config = {"name": "dice_loss", "params": {"smooth": 1.0}}
    loss_fn = factory.create_from_config(leaf_config)

    pred, target = sample_tensors
    result = loss_fn(pred, target)
    assert isinstance(result, torch.Tensor)

    # Test nested combination
    nested_config = {
        "type": "sum",
        "weights": [0.7, 0.3],
        "components": [
            {"name": "dice_loss", "params": {"smooth": 1.0}},
            {
                "type": "product",
                "components": [
                    {"name": "bce_loss"},
                    {"name": "dice_loss"},
                ],
            },
        ],
    }

    nested_loss = factory.create_from_config(nested_config)
    nested_result = nested_loss(pred, target)
    assert isinstance(nested_result, torch.Tensor)

    # Test configuration validation
    assert factory.validate_config(leaf_config)
    assert factory.validate_config(nested_config)

    # Test configuration summary
    summary = factory.get_config_summary(nested_config)
    assert summary["valid"]
    assert summary["depth"] == 3
    assert summary["leaf_count"] == 3
    assert summary["combinator_count"] == 2


def test_complete_integration_workflow(
    clean_registry: Any, sample_tensors: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Test the complete workflow from configuration to execution."""
    # Direct import using importlib to avoid circular dependencies
    import importlib.util
    import os

    factory_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "training",
        "losses",
        "factory",
        "recursive_factory.py",
    )
    factory_path = os.path.abspath(factory_path)

    spec = importlib.util.spec_from_file_location(
        "recursive_factory", factory_path
    )
    if spec is None:
        raise RuntimeError(f"Could not load spec for {factory_path}")
    factory_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(factory_module)

    RecursiveLossFactory = factory_module.RecursiveLossFactory

    # Create factory
    factory = RecursiveLossFactory()
    factory.registry = clean_registry

    # Complex configuration
    config = {
        "type": "sum",
        "weights": [0.5, 0.3, 0.2],
        "components": [
            {"name": "dice_loss", "params": {"smooth": 1.0}},
            {
                "type": "product",
                "components": [
                    {"name": "bce_loss"},
                    {"name": "dice_loss", "params": {"smooth": 2.0}},
                ],
            },
            {
                "type": "sum",
                "components": [
                    {"name": "dice_loss"},
                    {"name": "bce_loss"},
                ],
            },
        ],
    }

    # Create loss function
    loss_fn = factory.create_from_config(config)

    # Test execution
    pred, target = sample_tensors
    result = loss_fn(pred, target)

    assert isinstance(result, torch.Tensor)
    assert result.requires_grad
    assert result.dim() == 0  # Scalar loss

    # Test configuration summary
    summary = factory.get_config_summary(config)
    assert summary["valid"]
    assert summary["depth"] == 3
    assert summary["leaf_count"] == 6  # Total leaf losses in the configuration
    assert summary["combinator_count"] == 3  # Total combinators
