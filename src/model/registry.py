"""
Registry system for tracking and discovering model components.

Provides functionality to register, retrieve, and list available components
using a decorator pattern. Ensures type safety with generics.
"""

from typing import (Dict, List, Type, TypeVar, Generic, Callable, Optional,
                    Any, Set)


# Define a generic type for component base classes
T = TypeVar('T')


class Registry(Generic[T]):
    """
    Registry for model components with type safety.

    Allows registration, retrieval, and listing of model components of a
    specific base type. Uses decorators for easy registration.
    """

    def __init__(self, base_class: Type[T], name: str):
        """
        Initialize a Registry for components.

        Args:
            base_class (Type[T]): Base class that all registered components
                                 must inherit from.
            name (str): Name of the registry for identification.
        """
        self._base_class = base_class
        self._name = name
        self._components: Dict[str, Type[T]] = {}
        self._tags: Dict[str, Set[str]] = {}  # For component categorization

    def register(self, name: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> Callable[[Type[T]],
                                                               Type[T]]:
        """
        Decorator to register a component class in the registry.

        Args:
            name (Optional[str]): Name to register the component with.
                                 If None, uses the class name.
            tags (Optional[List[str]]): Tags for categorizing components.

        Returns:
            Callable: Decorator function that registers the component.

        Example:
            @encoder_registry.register()
            class ResNetEncoder(EncoderBase):
                pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            # Verify that the class inherits from the base class
            if not issubclass(cls, self._base_class):
                raise TypeError(
                    f"Class {cls.__name__} must inherit from "
                    f"{self._base_class.__name__}"
                )

            # Use provided name or class name
            component_name = name or cls.__name__

            # Check if component with this name already exists
            if component_name in self._components:
                raise ValueError(
                    f"Component '{component_name}' is already registered"
                )

            # Register the component
            self._components[component_name] = cls

            # Add tags if provided
            if tags:
                self._tags[component_name] = set(tags)
            else:
                self._tags[component_name] = set()

            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """
        Retrieve a component class by name.

        Args:
            name (str): Name of the component to retrieve.

        Returns:
            Type[T]: The registered component class.

        Raises:
            KeyError: If component is not found.
        """
        if name not in self._components:
            raise KeyError(
                f"Component '{name}' not found in registry '{self._name}'"
            )
        return self._components[name]

    def list(self) -> List[str]:
        """
        List all registered component names.

        Returns:
            List[str]: List of registered component names.
        """
        return list(self._components.keys())

    def list_with_tags(self) -> Dict[str, List[str]]:
        """
        List all registered components with their tags.

        Returns:
            Dict[str, List[str]]: Dictionary mapping component names to tags.
        """
        return {k: list(v) for k, v in self._tags.items()}

    def filter_by_tag(self, tag: str) -> List[str]:
        """
        Filter components by tag.

        Args:
            tag (str): Tag to filter by.

        Returns:
            List[str]: List of component names with the specified tag.
        """
        return [
            name for name, tags in self._tags.items() if tag in tags
        ]

    def instantiate(self, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Instantiate a registered component by name.

        Args:
            name (str): Name of the component to instantiate.
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            T: Instantiated component.

        Raises:
            KeyError: If component is not found.
        """
        component_cls = self.get(name)
        return component_cls(*args, **kwargs)

    @property
    def name(self) -> str:
        """Name of the registry."""
        return self._name

    @property
    def base_class(self) -> Type[T]:
        """Base class of the registry."""
        return self._base_class

    def __contains__(self, name: str) -> bool:
        """Check if a component is in the registry."""
        return name in self._components

    def __len__(self) -> int:
        """Get the number of registered components."""
        return len(self._components)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"{self._name} Registry with {len(self)} components"
