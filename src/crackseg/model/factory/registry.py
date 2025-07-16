"""
Registry system for tracking and discovering model components.

Provides functionality to register, retrieve, and list available components
using a decorator pattern. Ensures type safety with generics.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry[T]:
    """
    Thread-safe registry for model components with type safety.

    Allows registration, retrieval, and listing of model components of a
    specific base type. Uses decorators for easy registration.
    All operations are thread-safe.
    """

    def __init__(self, base_class: type[T], name: str) -> None:
        """
        Initialize a Registry for components.
        """
        self._base_class = base_class
        self._name = name
        self._components: dict[str, type[T]] = {}
        self._tags: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    #
    # Registration methods
    #
    def register(
        self, name: str | None = None, tags: list[str] | None = None
    ) -> Callable[[type[T]], type[T]]:
        """
        Thread-safe decorator to register a component class in the registry.

        Args:
            name: Name to register the component with.
                  If None, uses the class name.
            tags: Tags for categorizing components.

        Returns:
            Decorator function that registers the component.

        Example:
            @encoder_registry.register()
            class ResNetEncoder(EncoderBase):
                pass
        """

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, self._base_class):
                raise TypeError(
                    f"Class {cls.__name__} must inherit from "
                    f"{self._base_class.__name__}"
                )
            component_name: str = name if name is not None else cls.__name__
            with self._lock:
                if component_name in self._components:
                    raise ValueError(
                        f"Component '{component_name}' is already registered"
                    )
                self._components[component_name] = cls
                self._tags[component_name] = list(tags) if tags else []
            return cls

        return decorator

    def unregister(self, name: str) -> None:
        """
        Thread-safe removal of a component from the registry.

        Args:
            name: Name of the component to remove.

        Raises:
            KeyError: If component is not found.
        """
        with self._lock:
            if name not in self._components:
                raise KeyError(
                    f"Component '{name}' not found in registry '{self._name}'"
                )
            del self._components[name]
            # Also remove tags if present
            self._tags.pop(name, None)

    #
    # Component retrieval methods
    #
    def get(self, name: str) -> type[T]:
        """
        Retrieve a component class by name. Thread-safe.

        Args:
            name: Name of the component to retrieve.

        Returns:
            The registered component class (not an instance).

        Raises:
            KeyError: If component is not found.
        """
        with self._lock:
            if name not in self._components:
                raise KeyError(
                    f"Component '{name}' not found in registry '{self._name}'"
                )
            return self._components[name]

    def instantiate(self, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Instantiate a registered component by name. Thread-safe.

        Args:
            name: Name of the component to instantiate.
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            Instantiated component (instance of type T).

        Raises:
            KeyError: If component is not found.
        """
        component_cls = self.get(name)
        return component_cls(*args, **kwargs)

    #
    # Component listing and filtering methods
    #
    def list_components(self) -> list[str]:
        """
        List all registered component names. Thread-safe.

        Returns:
            List of registered component names.
        """
        with self._lock:
            return list(self._components.keys())

    def list_with_tags(self) -> dict[str, list[str]]:
        """
        List all registered components with their tags. Thread-safe.

        Returns:
            Dictionary mapping component names to their tags.
        """
        with self._lock:
            return {k: list(v) for k, v in self._tags.items()}

    def filter_by_tag(self, tag: str) -> list[str]:
        """
        Filter components by tag. Thread-safe.

        Args:
            tag: Tag to filter by.

        Returns:
            List of component names that have the specified tag.
        """
        with self._lock:
            return [name for name, tags in self._tags.items() if tag in tags]

    #
    # Properties and built-in method overrides
    #
    @property
    def name(self) -> str:
        """Name of the registry."""
        return self._name

    @property
    def base_class(self) -> type[T]:
        """Base class of the registry."""
        return self._base_class

    def __contains__(self, name: str) -> bool:
        """Check if a component is in the registry. Thread-safe."""
        with self._lock:
            return name in self._components

    def __len__(self) -> int:
        """Get the number of registered components. Thread-safe."""
        with self._lock:
            return len(self._components)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"Registry(name={self._name!r}, "
            f"base_class={self._base_class.__name__}, "
            f"components={list(self._components.keys())})"
        )
