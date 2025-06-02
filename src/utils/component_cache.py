"""
Component caching utilities for model instantiation.

This module provides a caching system for reusing instantiated model components
to improve performance during repeated instantiations.
"""

import hashlib
import weakref
from typing import Any, TypeVar

import torch.nn as nn

# Type variable for components
T = TypeVar("T", bound=nn.Module)

# Global cache using weak references to avoid memory leaks
_component_cache: weakref.WeakValueDictionary[str, nn.Module] = (
    weakref.WeakValueDictionary()
)


def generate_cache_key(component_type: str, config: dict[str, Any]) -> str:
    """
    Generate a unique cache key for a component configuration.

    Args:
        component_type: Type of the component (e.g., 'resnet_encoder')
        config: Configuration dictionary for the component

    Returns:
        Unique cache key string
    """
    # Create a deterministic string representation of the config
    config_str = str(sorted(config.items()))
    # Combine component type and config
    key_str = f"{component_type}:{config_str}"
    # Generate hash for consistent key length
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_component(cache_key: str) -> nn.Module | None:
    """
    Retrieve a component from the cache.

    Args:
        cache_key: Cache key for the component

    Returns:
        Cached component if found, None otherwise
    """
    return _component_cache.get(cache_key)


def cache_component(cache_key: str, component: nn.Module) -> None:
    """
    Store a component in the cache.

    Args:
        cache_key: Cache key for the component
        component: Component to cache
    """
    _component_cache[cache_key] = component


def clear_component_cache() -> None:
    """Clear all cached components."""
    _component_cache.clear()


def get_cache_info() -> dict[str, Any]:
    """
    Get information about the current cache state.

    Returns:
        Dictionary containing cache statistics
    """
    return {
        "cache_size": len(_component_cache),
        "cached_keys": list(_component_cache.keys()),
    }
