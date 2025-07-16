"""
Clean loss registry implementation that avoids circular dependencies.
"""

import importlib
from collections.abc import Callable
from typing import Any

from ..interfaces.loss_interface import ILossComponent


class RegistryError(Exception):
    """Base exception for registry-related errors."""

    pass


class LossNotFoundError(RegistryError):
    """Raised when trying to access a loss that is not registered."""

    pass


class LossAlreadyRegisteredError(RegistryError):
    """
    Raised when trying to register a loss with a name that already exists.
    """

    pass


class CleanLossRegistry:
    """
    Clean loss registry that uses lazy loading to avoid circular dependencies.

    This registry stores factory functions instead of importing classes
    directly, preventing import-time circular dependencies.
    """

    def __init__(self):
        self._factories: dict[str, Callable[..., ILossComponent]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register_factory(
        self,
        name: str,
        factory_func: Callable[..., ILossComponent],
        **metadata: Any,
    ) -> None:
        """
        Register a factory function for creating loss instances.

        Args:
            name: Unique name for the loss
            factory_func: Function that creates loss instances
            **metadata: Additional metadata about the loss
        """
        if name in self._factories:
            raise LossAlreadyRegisteredError(
                f"Loss '{name}' is already registered"
            )

        self._factories[name] = factory_func
        self._metadata[name] = metadata

    def register_class(
        self, name: str, module_path: str, class_name: str, **metadata: Any
    ) -> None:
        """
        Register a loss class using lazy loading.

        Args:
            name: Unique name for the loss
            module_path: Python module path
                (e.g., 'src.training.losses.dice_loss')
            class_name: Name of the class within the module
            **metadata: Additional metadata about the loss
        """

        def lazy_factory(**params: Any) -> ILossComponent:
            module = importlib.import_module(module_path)
            loss_class = getattr(module, class_name)
            return loss_class(**params)

        self.register_factory(name, lazy_factory, **metadata)

    def instantiate(self, name: str, **params: Any) -> ILossComponent:
        """
        Create an instance of a registered loss.

        Args:
            name: Name of the registered loss
            **params: Parameters to pass to the loss constructor

        Returns:
            Instance of the loss component

        Raises:
            LossNotFoundError: If the loss name is not registered
        """
        if name not in self._factories:
            available = list(self._factories.keys())
            raise LossNotFoundError(
                f"Loss '{name}' not found. Available losses: {available}"
            )

        try:
            return self._factories[name](**params)
        except Exception as e:
            raise RegistryError(
                f"Error instantiating loss '{name}' with params {params}: {e}"
            ) from e

    def is_registered(self, name: str) -> bool:
        """Check if a loss name is registered."""
        return name in self._factories

    def list_available(self) -> list[str]:
        """List all available registered loss names."""
        return list(self._factories.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered loss."""
        if name not in self._metadata:
            raise LossNotFoundError(f"Loss '{name}' not found")
        return self._metadata[name].copy()

    def register_decorator(
        self, name: str, **metadata: Any
    ) -> Callable[[type], type]:
        """
        Decorator for registering loss classes.

        Usage:
            @registry.register_decorator("my_loss", tags=["custom"])
            class MyLoss(SegmentationLoss):
                pass
        """

        def decorator(loss_class: type) -> type:
            def factory(**params: Any) -> ILossComponent:
                return loss_class(**params)

            self.register_factory(name, factory, **metadata)
            return loss_class

        return decorator
