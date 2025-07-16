"""
Configuration parser for nested loss structures.

This module provides functionality to parse, analyze, and transform loss
configuration dictionaries into structured data suitable for recursive loss
factory instantiation.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class ConfigParsingError(Exception):
    """Raised when configuration parsing fails."""

    pass


class ConfigNodeType:
    """Constants for configuration node types."""

    LEAF = "leaf"
    COMBINATOR = "combinator"


class ParsedNode:
    """
    Represents a parsed configuration node with metadata.
    """

    def __init__(
        self,
        node_type: str,
        config: dict[str, Any],
        children: list["ParsedNode"] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize a parsed node.

        Args:
            node_type: Type of node (leaf or combinator)
            config: Original configuration dictionary
            children: List of child nodes (for combinators)
            metadata: Additional metadata about the node
        """
        self.node_type = node_type
        self.config = config
        self.children = children or []
        self.metadata = metadata or {}

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type == ConfigNodeType.LEAF

    def is_combinator(self) -> bool:
        """Check if this is a combinator node."""
        return self.node_type == ConfigNodeType.COMBINATOR

    def get_combinator_type(self) -> str | None:
        """Get the combinator type if this is a combinator node."""
        if self.is_combinator():
            return self.config.get("type")
        return None

    def get_loss_name(self) -> str | None:
        """Get the loss name if this is a leaf node."""
        if self.is_leaf():
            return self.config.get("name")
        return None

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters for loss instantiation."""
        return cast(dict[str, Any], self.config.get("params", {}))

    def get_weights(self) -> list[float] | None:
        """Get weights for weighted combinators."""
        if self.is_combinator():
            return self.config.get("weights")
        return None


class ConfigParser:
    """
    Parses and analyzes nested loss configuration structures.
    """

    def __init__(self, validator: Any = None):
        """
        Initialize the configuration parser.

        Args:
            validator: Optional configuration validator
        """
        self.validator = validator
        self._parsing_errors: list[str] = []

    def parse(self, config: dict[str, Any]) -> ParsedNode:
        """
        Parse a configuration dictionary into a structured node tree.

        Args:
            config: Configuration dictionary to parse

        Returns:
            ParsedNode representing the root of the parsed tree

        Raises:
            ConfigParsingError: If parsing fails
        """
        self._parsing_errors.clear()

        try:
            # Validate configuration if validator is available
            if self.validator:
                self.validator.validate(config)

            # Parse the configuration tree
            root_node = self._parse_node(config, path="root")

            if self._parsing_errors:
                error_msg = "Parsing errors encountered:\n" + "\n".join(
                    self._parsing_errors
                )
                raise ConfigParsingError(error_msg)

            logger.info(
                f"Successfully parsed configuration with "
                f"{self._count_total_nodes(root_node)} nodes"
            )
            return root_node

        except Exception as e:
            logger.error(f"Configuration parsing failed: {e}")
            raise ConfigParsingError(
                f"Failed to parse configuration: {e}"
            ) from e

    def _parse_node(self, config: dict[str, Any], path: str) -> ParsedNode:
        """
        Recursively parse a configuration node.

        Args:
            config: Configuration node to parse
            path: Current path in configuration tree

        Returns:
            ParsedNode representing this configuration node
        """
        try:
            # Determine node type
            if self._is_combinator_node(config):
                return self._parse_combinator_node(config, path)
            elif self._is_leaf_node(config):
                return self._parse_leaf_node(config, path)
            else:
                error_msg = f"Unknown node type at {path}: {config}"
                self._parsing_errors.append(error_msg)
                raise ConfigParsingError(error_msg)

        except Exception as e:
            error_msg = f"Error parsing node at {path}: {e}"
            self._parsing_errors.append(error_msg)
            raise

    def _parse_combinator_node(
        self, config: dict[str, Any], path: str
    ) -> ParsedNode:
        """Parse a combinator configuration node."""
        logger.debug(f"Parsing combinator node at {path}")

        combinator_type = config.get("type")
        components = config.get("components", [])

        # Parse child components
        children = []
        for i, component_config in enumerate(components):
            child_path = f"{path}.components[{i}]"
            child_node = self._parse_node(component_config, child_path)
            children.append(child_node)

        # Extract and validate weights
        weights = self._extract_and_validate_weights(
            config, cast(list[ParsedNode], children), path
        )

        # Create metadata
        metadata = {
            "combinator_type": combinator_type,
            "component_count": len(cast(list[ParsedNode], children)),
            "has_weights": weights is not None,
            "path": path,
        }

        # Update config with normalized weights if applicable
        processed_config = config.copy()
        if weights is not None:
            processed_config["weights"] = weights

        return ParsedNode(
            node_type=ConfigNodeType.COMBINATOR,
            config=processed_config,
            children=cast(list[ParsedNode], children),
            metadata=metadata,
        )

    def _parse_leaf_node(
        self, config: dict[str, Any], path: str
    ) -> ParsedNode:
        """Parse a leaf loss configuration node."""
        logger.debug(f"Parsing leaf node at {path}")

        loss_name = config.get("name")
        parameters = config.get("params", {})

        # Create metadata
        metadata = {
            "loss_name": loss_name,
            "parameter_count": len(parameters),
            "path": path,
        }

        return ParsedNode(
            node_type=ConfigNodeType.LEAF,
            config=config,
            children=[],
            metadata=metadata,
        )

    def _extract_and_validate_weights(
        self, config: dict[str, Any], children: list[ParsedNode], path: str
    ) -> list[float] | None:
        """
        Extract and validate weights for weighted combinators.

        Args:
            config: Combinator configuration
            children: List of child nodes
            path: Current path in configuration

        Returns:
            Normalized weights list or None if no weights specified
        """
        combinator_type = config.get("type")
        weights = config.get("weights")

        # Only process weights for sum combinators
        if combinator_type != "sum":
            return None

        # If no weights specified, use equal weights
        if weights is None:
            num_components = len(children)
            equal_weight = 1.0 / num_components
            weights = [equal_weight] * num_components
            logger.debug(f"Using equal weights at {path}: {weights}")
            return weights

        # Validate weights
        if not isinstance(weights, list):
            error_msg = f"Weights must be a list at {path}"
            self._parsing_errors.append(error_msg)
            return None

        if len(cast(list[float], weights)) != len(children):
            error_msg = (
                f"Number of weights ({len(cast(list[float], weights))}) must "
                f"match number of components ({len(children)}) at {path}"
            )
            self._parsing_errors.append(error_msg)
            return None

        # Check for positive weights
        if any(w <= 0 for w in weights):
            error_msg = f"All weights must be positive at {path}"
            self._parsing_errors.append(error_msg)
            return None

        # Normalize weights
        normalized_weights = self._normalize_weights(weights)
        logger.debug(f"Normalized weights at {path}: {normalized_weights}")
        return normalized_weights

    def _normalize_weights(self, weights: list[float]) -> list[float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: List of weights to normalize

        Returns:
            Normalized weights list
        """
        total = sum(weights)
        if total == 0:
            raise ConfigParsingError("Cannot normalize weights: sum is zero")

        return [w / total for w in weights]

    def _is_combinator_node(self, config: dict[str, Any]) -> bool:
        """Check if configuration represents a combinator node."""
        return "type" in config and config["type"] in {"sum", "product"}

    def _is_leaf_node(self, config: dict[str, Any]) -> bool:
        """Check if configuration represents a leaf loss node."""
        return "name" in config

    def _count_total_nodes(self, node: ParsedNode) -> int:
        """Count total nodes in the parsed tree."""
        count = 1
        for child in node.children:
            count += self._count_total_nodes(child)
        return count

    def get_parsing_errors(self) -> list[str]:
        """
        Get list of parsing errors encountered during last parse operation.
        """
        return self._parsing_errors.copy()

    def analyze_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze a configuration and return structural information.

        Args:
            config: Configuration to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            parsed_tree = self.parse(config)
            return self._analyze_parsed_tree(parsed_tree)
        except ConfigParsingError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "analysis_failed": True,
            }

    def _analyze_parsed_tree(self, root: ParsedNode) -> dict[str, Any]:
        """Analyze a parsed configuration tree."""
        analysis = {
            "valid": True,
            "total_nodes": self._count_total_nodes(root),
            "leaf_count": self._count_leaf_nodes(root),
            "combinator_count": self._count_combinator_nodes(root),
            "max_depth": self._calculate_max_depth(root),
            "combinator_types": self._get_combinator_types(root),
            "loss_types": self._get_loss_types(root),
            "weighted_combinators": self._count_weighted_combinators(root),
        }

        return analysis

    def _count_leaf_nodes(self, node: ParsedNode) -> int:
        """Count leaf nodes in the tree."""
        if node.is_leaf():
            return 1

        count = 0
        for child in node.children:
            count += self._count_leaf_nodes(child)
        return count

    def _count_combinator_nodes(self, node: ParsedNode) -> int:
        """Count combinator nodes in the tree."""
        count = 1 if node.is_combinator() else 0
        for child in node.children:
            count += self._count_combinator_nodes(child)
        return count

    def _calculate_max_depth(self, node: ParsedNode) -> int:
        """Calculate maximum depth of the tree."""
        if not node.children:
            return 1

        max_child_depth = max(
            self._calculate_max_depth(child) for child in node.children
        )
        return 1 + max_child_depth

    def _get_combinator_types(self, node: ParsedNode) -> list[str]:
        """Get list of all combinator types used in the tree."""
        types = []
        if node.is_combinator():
            types.append(node.get_combinator_type())

        for child in node.children:
            types.extend(self._get_combinator_types(child))

        return list({cast(str, t) for t in types if t is not None})

    def _get_loss_types(self, node: ParsedNode) -> list[str]:
        """Get list of all loss types used in the tree."""
        types = []
        if node.is_leaf():
            types.append(node.get_loss_name())

        for child in node.children:
            types.extend(self._get_loss_types(child))

        return list({cast(str, t) for t in types if t is not None})

    def _count_weighted_combinators(self, node: ParsedNode) -> int:
        """Count combinators that use explicit weights."""
        count = 0
        if node.is_combinator() and node.get_weights() is not None:
            # Check if weights were explicitly provided (not auto-generated
            # equal weights)
            original_config = node.config
            if "weights" in original_config:
                count = 1

        for child in node.children:
            count += self._count_weighted_combinators(child)

        return count
