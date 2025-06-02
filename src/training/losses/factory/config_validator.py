"""
Configuration validator for loss factory.
"""

from typing import Any, cast


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """
    Validates loss configuration dictionaries for correctness and completeness.
    """

    def __init__(self, registry: Any):
        """
        Initialize validator with registry reference.

        Args:
            registry: Loss registry for checking available losses
        """
        self.registry = registry

    def validate(self, config: dict[str, Any]) -> None:
        """
        Validate a loss configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        visited_nodes: set[int] = set()
        self._validate_node(config, visited_nodes, path="root")

    def _validate_node(
        self, node: dict[str, Any], visited: set[int], path: str
    ) -> None:
        """
        Recursively validate a configuration node.

        Args:
            node: Configuration node to validate
            visited: Set of visited node IDs (for cycle detection)
            path: Current path in configuration tree (for error messages)
        """
        # Check for cycles
        node_id = id(node)
        if node_id in visited:
            raise ConfigValidationError(
                f"Circular reference detected at {path}"
            )
        visited.add(node_id)

        try:
            if "type" in node:
                self._validate_combinator_node(node, visited, path)
            elif "name" in node:
                self._validate_leaf_node(node, path)
            else:
                raise ConfigValidationError(
                    f"Invalid node at {path}: must have either 'type' or "
                    "'name'"
                )
        finally:
            visited.remove(node_id)

    def _validate_combinator_node(
        self, node: dict[str, Any], visited: set[int], path: str
    ) -> None:
        """Validate a combinator node."""
        comb_type = node["type"]

        # Validate combinator type
        valid_types = {"sum", "product"}
        if comb_type not in valid_types:
            raise ConfigValidationError(
                f"Invalid combinator type '{comb_type}' at {path}. "
                f"Valid types: {valid_types}"
            )

        # Validate components
        if "components" not in node:
            raise ConfigValidationError(
                f"Missing 'components' in combinator at {path}"
            )

        components = node["components"]
        if not isinstance(components, list) or not components:
            raise ConfigValidationError(
                f"'components' must be a non-empty list at {path}"
            )

        # Validate weights for sum combinators
        if comb_type == "sum" and "weights" in node:
            weights = node["weights"]
            if not isinstance(weights, list):
                raise ConfigValidationError(
                    f"'weights' must be a list at {path}"
                )
            if len(cast(list[float], weights)) != len(
                cast(list[dict[str, Any]], components)
            ):
                raise ConfigValidationError(
                    f"Number of weights ({len(cast(list[float], weights))}) "
                    "must match number of components "
                    f"({len(cast(list[dict[str, Any]], components))}) at "
                    f"{path}"
                )
            if any(w <= 0 for w in cast(list[float], weights)):
                raise ConfigValidationError(
                    f"All weights must be positive at {path}"
                )

        # Recursively validate components
        for i, component in enumerate(cast(list[dict[str, Any]], components)):
            self._validate_node(component, visited, f"{path}.components[{i}]")

    def _validate_leaf_node(self, node: dict[str, Any], path: str) -> None:
        """Validate a leaf loss node."""
        loss_name = node["name"]

        # Check if loss is registered
        if not self.registry.is_registered(loss_name):
            available = self.registry.list_available()
            raise ConfigValidationError(
                f"Loss '{loss_name}' not registered at {path}. "
                f"Available losses: {available}"
            )

        # Validate params if present
        if "params" in node:
            params = node["params"]
            if not isinstance(params, dict):
                raise ConfigValidationError(
                    f"'params' must be a dictionary at {path}"
                )

    def get_validation_summary(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Get a summary of configuration validation results.

        Args:
            config: Configuration to analyze

        Returns:
            Dictionary with validation summary
        """
        try:
            self.validate(config)
            return {
                "valid": True,
                "depth": self._calculate_depth(config),
                "leaf_count": self._count_leaves(config),
                "combinator_count": self._count_combinators(config),
                "errors": [],
            }
        except ConfigValidationError as e:
            return {"valid": False, "errors": [str(e)]}

    def _calculate_depth(self, node: dict[str, Any]) -> int:
        """Calculate maximum depth of configuration tree."""
        if "type" in node and "components" in node:
            return 1 + max(
                self._calculate_depth(comp)
                for comp in cast(list[dict[str, Any]], node["components"])
            )
        return 1

    def _count_leaves(self, node: dict[str, Any]) -> int:
        """Count leaf nodes in configuration tree."""
        if "name" in node:
            return 1
        elif "type" in node and "components" in node:
            return sum(
                self._count_leaves(comp)
                for comp in cast(list[dict[str, Any]], node["components"])
            )
        return 0

    def _count_combinators(self, node: dict[str, Any]) -> int:
        """Count combinator nodes in configuration tree."""
        if "type" in node:
            count = 1
            if "components" in node:
                count += sum(
                    self._count_combinators(comp)
                    for comp in cast(list[dict[str, Any]], node["components"])
                )
            return count
        return 0
