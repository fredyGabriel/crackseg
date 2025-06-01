"""
Unit tests for the configuration parser.
"""

from typing import Any

import pytest
import torch

from src.training.losses.factory.config_parser import (
    ConfigNodeType,
    ConfigParser,
    ConfigParsingError,
    ParsedNode,
)
from src.training.losses.factory.config_validator import ConfigValidator
from src.training.losses.registry.clean_registry import CleanLossRegistry


@pytest.fixture
def mock_registry() -> CleanLossRegistry:
    """Create a mock registry for testing."""
    registry = CleanLossRegistry()

    # Register mock losses
    def mock_dice_loss(**params: Any) -> torch.nn.Module:
        class MockDiceLoss(torch.nn.Module):
            def __init__(self, smooth: float = 1.0) -> None:
                super().__init__()  # type: ignore
                self.smooth = smooth

        p = dict(params)
        if "smooth" in p:
            p["smooth"] = float(p["smooth"])
        return MockDiceLoss(**p)

    def mock_bce_loss(**params: Any) -> torch.nn.Module:
        class MockBCELoss(torch.nn.Module):
            def __init__(self, reduction: str = "mean") -> None:
                super().__init__()  # type: ignore
                self.reduction = reduction

        p = dict(params)
        if "reduction" in p:
            p["reduction"] = str(p["reduction"])
        return MockBCELoss(**p)

    registry.register_factory("dice_loss", mock_dice_loss)  # type: ignore
    registry.register_factory("bce_loss", mock_bce_loss)  # type: ignore

    return registry


@pytest.fixture
def parser(mock_registry: CleanLossRegistry) -> ConfigParser:
    """Create a parser with mock registry."""
    validator = ConfigValidator(mock_registry)
    return ConfigParser(validator)


class TestParsedNode:
    """Test the ParsedNode class."""

    def test_leaf_node_creation(self) -> None:
        """Test creating a leaf node."""
        config = {"name": "dice_loss", "params": {"smooth": 1.0}}
        node = ParsedNode(ConfigNodeType.LEAF, config)

        assert node.is_leaf()
        assert not node.is_combinator()
        assert node.get_loss_name() == "dice_loss"
        assert node.get_parameters() == {"smooth": 1.0}
        assert node.get_combinator_type() is None
        assert node.get_weights() is None

    def test_combinator_node_creation(self) -> None:
        """Test creating a combinator node."""
        config = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }
        children = [
            ParsedNode(ConfigNodeType.LEAF, {"name": "dice_loss"}),
            ParsedNode(ConfigNodeType.LEAF, {"name": "bce_loss"}),
        ]
        node = ParsedNode(ConfigNodeType.COMBINATOR, config, children)

        assert not node.is_leaf()
        assert node.is_combinator()
        assert node.get_combinator_type() == "sum"
        assert node.get_weights() == [0.6, 0.4]
        assert node.get_loss_name() is None
        assert len(node.children) == 2


class TestConfigParser:
    """Test the ConfigParser class."""

    def test_parse_simple_leaf_config(self, parser: ConfigParser) -> None:
        """Test parsing a simple leaf configuration."""
        config = {"name": "dice_loss", "params": {"smooth": 1.5}}
        parsed = parser.parse(config)
        assert parsed.is_leaf()
        assert parsed.get_loss_name() == "dice_loss"
        assert parsed.get_parameters() == {"smooth": 1.5}
        assert len(parsed.children) == 0

    def test_parse_simple_combinator_config(
        self, parser: ConfigParser
    ) -> None:
        """Test parsing a simple combinator configuration."""
        config = {
            "type": "sum",
            "weights": [0.7, 0.3],
            "components": [
                {"name": "dice_loss", "params": {"smooth": 1.0}},
                {"name": "bce_loss"},
            ],
        }
        parsed = parser.parse(config)
        assert parsed.is_combinator()
        assert parsed.get_combinator_type() == "sum"
        assert parsed.get_weights() == [0.7, 0.3]
        assert len(parsed.children) == 2
        child1, child2 = parsed.children
        assert child1.is_leaf()
        assert child1.get_loss_name() == "dice_loss"
        assert child1.get_parameters() == {"smooth": 1.0}
        assert child2.is_leaf()
        assert child2.get_loss_name() == "bce_loss"
        assert child2.get_parameters() == {}

    def test_parse_nested_configuration(self, parser: ConfigParser) -> None:
        """Test parsing a nested configuration."""
        config = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {
                    "type": "product",
                    "components": [
                        {"name": "bce_loss"},
                        {"name": "dice_loss", "params": {"smooth": 2.0}},
                    ],
                },
            ],
        }
        parsed = parser.parse(config)
        assert parsed.is_combinator()
        assert parsed.get_combinator_type() == "sum"
        assert len(parsed.children) == 2
        leaf_child = parsed.children[0]
        assert leaf_child.is_leaf()
        assert leaf_child.get_loss_name() == "dice_loss"
        combinator_child = parsed.children[1]
        assert combinator_child.is_combinator()
        assert combinator_child.get_combinator_type() == "product"
        assert len(combinator_child.children) == 2
        grandchild1, grandchild2 = combinator_child.children
        assert grandchild1.is_leaf()
        assert grandchild1.get_loss_name() == "bce_loss"
        assert grandchild2.is_leaf()
        assert grandchild2.get_loss_name() == "dice_loss"
        assert grandchild2.get_parameters() == {"smooth": 2.0}

    def test_weight_normalization(self, parser: ConfigParser) -> None:
        """Test automatic weight normalization."""
        config = {
            "type": "sum",
            "weights": [3.0, 6.0, 1.0],  # Should normalize to [0.3, 0.6, 0.1]
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
                {"name": "dice_loss"},
            ],
        }

        parsed = parser.parse(config)
        weights = parsed.get_weights()

        expected_weights = [0.3, 0.6, 0.1]
        if weights is not None:
            assert len(weights) == 3
            for actual, expected in zip(
                weights, expected_weights, strict=False
            ):
                assert abs(actual - expected) < 1e-6
        else:
            pytest.fail("weights should not be None")

    def test_equal_weights_for_sum_without_weights(
        self, parser: ConfigParser
    ) -> None:
        """Test that equal weights are assigned when not specified for sum
        combinator."""
        config = {
            "type": "sum",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
                {"name": "dice_loss"},
            ],
        }

        parsed = parser.parse(config)
        weights = parsed.get_weights()

        expected_weight = 1.0 / 3.0
        if weights is not None:
            assert len(weights) == 3
            for weight in weights:
                assert abs(weight - expected_weight) < 1e-6
        else:
            pytest.fail("weights should not be None")

    def test_product_combinator_no_weights(self, parser: ConfigParser) -> None:
        """Test that product combinators don't have weights."""
        config = {
            "type": "product",
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        parsed = parser.parse(config)

        assert parsed.get_combinator_type() == "product"
        assert parsed.get_weights() is None

    def test_parsing_errors_collection(self, parser: ConfigParser) -> None:
        """Test that parsing errors are collected properly."""
        config = {
            "type": "sum",
            "weights": [0.5],  # Wrong number of weights
            "components": [
                {"name": "dice_loss"},
                {"name": "bce_loss"},
            ],
        }

        with pytest.raises(ConfigParsingError) as exc_info:
            parser.parse(config)

        assert "Number of weights" in str(exc_info.value)

    def test_invalid_node_type(self, parser: ConfigParser) -> None:
        """Test parsing fails for invalid node types."""
        config = {"invalid": "config"}

        with pytest.raises(ConfigParsingError) as exc_info:
            parser.parse(config)

        assert "Unknown node type" in str(exc_info.value)

    def test_analyze_configuration(self, parser: ConfigParser) -> None:
        """Test configuration analysis functionality."""
        config = {
            "type": "sum",
            "weights": [0.6, 0.4],
            "components": [
                {"name": "dice_loss"},
                {
                    "type": "product",
                    "components": [
                        {"name": "bce_loss"},
                        {"name": "dice_loss"},
                    ],
                },
            ],
        }

        analysis = parser.analyze_configuration(config)

        assert analysis["valid"] is True
        assert (
            analysis["total_nodes"] == 5
        )  # 1 root + 1 leaf + 1 combinator + 2 leaves
        assert analysis["leaf_count"] == 3
        assert analysis["combinator_count"] == 2
        assert analysis["max_depth"] == 3
        assert set(analysis["combinator_types"]) == {"sum", "product"}
        assert set(analysis["loss_types"]) == {"dice_loss", "bce_loss"}

    def test_parser_without_validator(self):
        """Test parser works without validator."""
        parser = ConfigParser()
        config = {"name": "dice_loss"}

        # Should not raise validation errors since no validator
        parsed = parser.parse(config)
        assert parsed.is_leaf()
        assert parsed.get_loss_name() == "dice_loss"
