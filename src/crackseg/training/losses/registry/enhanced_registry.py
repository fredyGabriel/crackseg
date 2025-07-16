"""
Enhanced loss registry with advanced instantiation features.

This module extends the basic registry with caching, parameter validation,
type checking, and improved error handling for production use.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

from ..interfaces.loss_interface import ILossComponent
from .clean_registry import CleanLossRegistry, LossNotFoundError, RegistryError

logger = logging.getLogger(__name__)


class ParameterValidationError(RegistryError):
    """Raised when loss parameters fail validation."""

    pass


class TypeValidationError(RegistryError):
    """Raised when instantiated loss doesn't match expected type."""

    pass


class EnhancedLossRegistry(CleanLossRegistry):
    """
    Enhanced loss registry with caching, validation, and type checking.

    Features:
    - Instance caching for repeated loss types with same parameters
    - Parameter validation against function signatures
    - Type checking for instantiated losses
    - Enhanced error messages with suggestions
    - Dynamic module loading support
    - Performance monitoring and statistics
    """

    def __init__(
        self, enable_caching: bool = True, cache_size_limit: int = 100
    ):
        """
        Initialize enhanced registry.

        Args:
            enable_caching: Whether to enable instance caching
            cache_size_limit: Maximum number of cached instances
        """
        super().__init__()
        self.enable_caching = enable_caching
        self.cache_size_limit = cache_size_limit

        # Instance cache: (loss_name, params_hash) -> loss_instance
        self._instance_cache: dict[
            tuple[str, tuple[tuple[str, Any], ...]], ILossComponent
        ] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Parameter schemas for validation
        self._parameter_schemas: dict[str, dict[str, Any]] = {}

        # Type constraints
        self._type_constraints: dict[str, type] = {}

        # Usage statistics
        self._instantiation_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}

    def register_factory(
        self,
        name: str,
        factory_func: Callable[..., ILossComponent],
        parameter_schema: dict[str, Any] | None = None,
        expected_type: type | None = None,
        **metadata: Any,
    ) -> None:
        """
        Register a factory function with enhanced validation.

        Args:
            name: Unique name for the loss
            factory_func: Function that creates loss instances
            parameter_schema: Schema for parameter validation
            expected_type: Expected type/interface for instantiated losses
            **metadata: Additional metadata about the loss
        """
        super().register_factory(name, factory_func, **metadata)

        # Store parameter schema
        if parameter_schema:
            self._parameter_schemas[name] = parameter_schema
        else:
            # Auto-generate schema from function signature
            self._parameter_schemas[name] = self._extract_parameter_schema(
                factory_func
            )

        # Store type constraint
        if expected_type:
            self._type_constraints[name] = expected_type

        logger.debug(f"Registered loss '{name}' with enhanced validation")

    def instantiate(self, name: str, **params: Any) -> ILossComponent:
        """
        Create an instance of a registered loss with enhanced validation.

        Args:
            name: Name of the registered loss
            **params: Parameters to pass to the loss constructor

        Returns:
            Instance of the loss component

        Raises:
            LossNotFoundError: If the loss name is not registered
            ParameterValidationError: If parameters are invalid
            TypeValidationError: If instantiated loss doesn't match expected
            type
        """
        # Check if loss is registered
        if name not in self._factories:
            self._error_counts[name] = self._error_counts.get(name, 0) + 1
            available = self._get_similar_names(name)
            suggestion = f" Did you mean: {available}?" if available else ""
            raise LossNotFoundError(
                f"Loss '{name}' not found.{suggestion} "
                f"Available losses: {self.list_available()}"
            )

        # Validate parameters
        self._validate_parameters(name, params)

        # Check cache first
        cache_key: tuple[str, tuple[tuple[str, Any], ...]] | None = None
        if self.enable_caching:
            cache_key = self._create_cache_key(name, params)
            if cache_key in self._instance_cache:
                self._cache_hits += 1
                logger.debug(
                    f"Cache hit for loss '{name}' with params {params}"
                )
                return self._instance_cache[cache_key]
            else:
                self._cache_misses += 1

        # Instantiate loss
        try:
            instance = self._factories[name](**params)
            self._instantiation_counts[name] = (
                self._instantiation_counts.get(name, 0) + 1
            )

            # Validate instance type
            self._validate_instance_type(name, instance)

            # Cache instance if enabled
            if self.enable_caching:
                assert cache_key is not None
                self._cache_instance(cache_key, instance)

            logger.debug(
                f"Successfully instantiated loss '{name}' with params {params}"
            )
            return instance

        except Exception as e:
            self._error_counts[name] = self._error_counts.get(name, 0) + 1
            raise RegistryError(
                f"Error instantiating loss '{name}' with params {params}: "
                f"{e}. Check parameter types and values."
            ) from e

    def _validate_parameters(self, name: str, params: dict[str, Any]) -> None:
        """Validate parameters against schema."""
        if name not in self._parameter_schemas:
            return  # No schema available, skip validation

        schema = self._parameter_schemas[name]

        # Check required parameters
        required_params = schema.get("required", set())
        missing_params = required_params - set(params.keys())
        if missing_params:
            raise ParameterValidationError(
                f"Missing required parameters for loss '{name}': "
                f"{missing_params}"
            )

        # Check parameter types
        param_types = schema.get("types", {})
        for param_name, param_value in params.items():
            if param_name in param_types:
                expected_type = param_types[param_name]
                if not isinstance(param_value, expected_type):
                    raise ParameterValidationError(
                        f"Parameter '{param_name}' for loss '{name}' must be "
                        f"of type {expected_type.__name__}, got "
                        f"{type(param_value).__name__}"
                    )

        # Check parameter ranges/constraints
        constraints = schema.get("constraints", {})
        for param_name, param_value in params.items():
            if param_name in constraints:
                constraint = constraints[param_name]
                if not self._check_constraint(param_value, constraint):
                    raise ParameterValidationError(
                        f"Parameter '{param_name}' for loss '{name}' violates "
                        f"constraint: {constraint}"
                    )

    def _validate_instance_type(self, name: str, instance: Any) -> None:
        """Validate that instantiated loss matches expected type."""
        if name not in self._type_constraints:
            return  # No type constraint, skip validation

        expected_type = self._type_constraints[name]
        if not isinstance(instance, expected_type):
            raise TypeValidationError(
                f"Loss '{name}' instantiated to {type(instance).__name__}, "
                f"expected {expected_type.__name__}"
            )

    def _extract_parameter_schema(
        self, factory_func: Callable[..., Any]
    ) -> dict[str, Any]:
        """Extract parameter schema from function signature."""
        try:
            sig = inspect.signature(factory_func)
            schema: dict[str, Any] = {
                "required": set(),
                "types": {},
                "defaults": {},
            }

            for param_name, param in sig.parameters.items():
                # Skip self and *args, **kwargs
                if param_name in ("self", "args", "kwargs"):
                    continue

                # Check if required (no default value)
                if param.default == inspect.Parameter.empty:
                    schema["required"].add(param_name)
                else:
                    schema["defaults"][param_name] = param.default

                # Extract type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    schema["types"][param_name] = param.annotation

            return schema

        except Exception as e:
            logger.warning(
                f"Failed to extract schema from {factory_func}: {e}"
            )
            return {}

    def _create_cache_key(
        self, name: str, params: dict[str, Any]
    ) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Create a hashable cache key from name and parameters."""
        # Sort parameters for consistent hashing
        sorted_params = tuple(sorted(params.items()))
        return (name, sorted_params)

    def _cache_instance(
        self,
        cache_key: tuple[str, tuple[tuple[str, Any], ...]],
        instance: ILossComponent,
    ) -> None:
        """Cache an instance with size limit management."""
        if len(self._instance_cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._instance_cache))
            del self._instance_cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")

        self._instance_cache[cache_key] = instance

    def _get_similar_names(
        self, name: str, max_suggestions: int = 3
    ) -> list[str]:
        """Get similar loss names for suggestions."""
        available = self.list_available()

        # Simple similarity based on common prefixes/suffixes
        similar: list[str] = []
        name_lower = name.lower()

        for available_name in available:
            available_lower = available_name.lower()

            # Check for common substrings
            if (
                name_lower in available_lower
                or available_lower in name_lower
                or self._levenshtein_distance(name_lower, available_lower) <= 2
            ):
                similar.append(available_name)

        return similar[:max_suggestions]

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _check_constraint(
        self, value: Any, constraint: dict[str, Any]
    ) -> bool:
        """Check if value satisfies constraint."""
        if "min" in constraint and value < constraint["min"]:
            return False
        if "max" in constraint and value > constraint["max"]:
            return False
        if "choices" in constraint and value not in constraint["choices"]:
            return False
        return True

    def clear_cache(self) -> None:
        """Clear the instance cache."""
        self._instance_cache.clear()
        logger.info("Cleared loss instance cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests if total_requests > 0 else 0
        )

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_instances": len(self._instance_cache),
            "cache_size_limit": self.cache_size_limit,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "instantiation_counts": self._instantiation_counts.copy(),
            "error_counts": self._error_counts.copy(),
            "total_instantiations": sum(self._instantiation_counts.values()),
            "total_errors": sum(self._error_counts.values()),
        }

    def register_from_module(
        self,
        module_path: str,
        loss_classes: list[str] | None = None,
        name_prefix: str = "",
    ) -> None:
        """
        Dynamically register losses from a module.

        Args:
            module_path: Python module path to import
            loss_classes: Specific class names to register (None = all)
            name_prefix: Prefix to add to registered names
        """
        try:
            import importlib

            module = importlib.import_module(module_path)

            # Get all classes if not specified
            if loss_classes is None:
                loss_classes = [
                    name
                    for name in dir(module)
                    if (
                        isinstance(getattr(module, name), type)
                        and name.endswith("Loss")
                    )
                ]

            for class_name in loss_classes:
                if hasattr(module, class_name):
                    loss_class = getattr(module, class_name)
                    registered_name = f"{name_prefix}{class_name.lower()}"

                    def factory(
                        loss_class: type = loss_class, **params: Any
                    ) -> ILossComponent:
                        return loss_class(**params)

                    self.register_factory(
                        registered_name,
                        factory,
                        module_path=module_path,
                        class_name=class_name,
                    )

                    logger.info(
                        f"Dynamically registered '{registered_name}' from "
                        f"{module_path}"
                    )

        except Exception as e:
            logger.error(f"Failed to register losses from {module_path}: {e}")
            raise RegistryError(f"Dynamic registration failed: {e}") from e
