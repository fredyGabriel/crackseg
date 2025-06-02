"""
Recursive loss factory implementation using clean architecture.
"""

from typing import Any

from ..combinators import ProductCombinator, WeightedSumCombinator
from ..interfaces.loss_interface import ILossComponent
from ..registry import registry
from .config_parser import ConfigParser, ConfigParsingError, ParsedNode
from .config_validator import ConfigValidationError, ConfigValidator


class RecursiveFactoryError(Exception):
    """Base exception for recursive factory errors."""

    pass


class RecursiveLossFactory:
    """
    Clean recursive loss factory that builds loss hierarchies from
    configuration.

    This factory uses dependency injection and lazy loading to avoid circular
    dependencies while supporting arbitrary nesting of loss combinations.
    """

    def __init__(self):
        """Initialize factory with default registry and validator."""
        self.registry = registry
        self.validator = ConfigValidator(self.registry)
        self.parser = ConfigParser(self.validator)

    def create_from_config(self, config: dict[str, Any]) -> ILossComponent:
        """
        Create a loss component hierarchy from configuration.

        Args:
            config: Configuration dictionary defining the loss hierarchy

        Returns:
            Loss component (either leaf loss or combinator)

        Raises:
            RecursiveFactoryError: If configuration is invalid or creation
            fails
        """
        try:
            # Parse and validate configuration
            parsed_tree = self.parser.parse(config)

            # Build component hierarchy from parsed tree
            return self._build_from_parsed_node(parsed_tree)

        except (ConfigValidationError, ConfigParsingError) as e:
            raise RecursiveFactoryError(
                f"Configuration processing failed: {e}"
            ) from e
        except Exception as e:
            raise RecursiveFactoryError(
                f"Failed to create loss component: {e}"
            ) from e

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate configuration without creating components.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.parser.parse(config)
            return True
        except (ConfigValidationError, ConfigParsingError):
            return False

    def get_config_summary(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Get detailed summary of configuration structure.

        Args:
            config: Configuration to analyze

        Returns:
            Dictionary with configuration analysis
        """
        return self.parser.analyze_configuration(config)

    def _build_from_parsed_node(self, node: ParsedNode) -> ILossComponent:
        """
        Build a loss component from a parsed configuration node.

        Args:
            node: Parsed configuration node

        Returns:
            Loss component instance
        """
        if node.is_leaf():
            return self._build_leaf_loss(node)
        elif node.is_combinator():
            return self._build_combinator(node)
        else:
            raise RecursiveFactoryError(f"Unknown node type: {node.node_type}")

    def _build_combinator(self, node: ParsedNode) -> ILossComponent:
        """Build a combinator component from parsed node."""
        comb_type = node.get_combinator_type()

        # Recursively build child components
        components = [
            self._build_from_parsed_node(child) for child in node.children
        ]

        # Create appropriate combinator
        if comb_type == "sum":
            weights = node.get_weights()
            return WeightedSumCombinator(components, weights)
        elif comb_type == "product":
            return ProductCombinator(components)
        else:
            raise RecursiveFactoryError(
                f"Unsupported combinator type: {comb_type}"
            )

    def _build_leaf_loss(self, node: ParsedNode) -> ILossComponent:
        """Build a leaf loss component from parsed node."""
        loss_name = node.get_loss_name()
        if loss_name is None:
            raise RecursiveFactoryError("Loss name is missing in leaf node")
        params = node.get_parameters()

        try:
            return self.registry.instantiate(loss_name, **params)
        except Exception as e:
            raise RecursiveFactoryError(
                f"Failed to instantiate loss '{loss_name}' with params "
                f"{params}: {e}"
            ) from e

    # Legacy methods for backward compatibility
    def _build_component(self, config: dict[str, Any]) -> ILossComponent:
        """
        Legacy method for backward compatibility.
        Use create_from_config instead.
        """
        return self.create_from_config(config)


# Convenience function for direct usage
def create_loss_from_config(config: dict[str, Any]) -> ILossComponent:
    """
    Convenience function to create loss from configuration.

    Args:
        config: Loss configuration dictionary

    Returns:
        Loss component instance
    """
    factory = RecursiveLossFactory()
    return factory.create_from_config(config)
