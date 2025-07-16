"""
Component Caching Utilities.

Provides functions to cache and retrieve instantiated PyTorch modules
using weak references to avoid memory leaks. Includes cache clearing
and key generation logic.
"""

import logging
import weakref
from typing import Any, cast

from torch import nn

# Create logger
log = logging.getLogger(__name__)

# Component cache system (using weak references)
_component_cache: dict[str, weakref.ReferenceType[nn.Module]] = {}


def clear_component_cache() -> None:
    """Clear the component cache."""
    _component_cache.clear()
    log.info("Component cache cleared")


def get_cached_component(cache_key: str) -> nn.Module | None:
    """Retrieve a component from cache if available."""
    if cache_key in _component_cache:
        component_ref = _component_cache[cache_key]
        component = component_ref()
        if component is not None:
            log.debug(f"Cache hit for component: {cache_key}")
            return component
        else:
            # Reference has been garbage collected
            del _component_cache[cache_key]
            log.debug(f"Removed expired cache entry: {cache_key}")
    return None


def cache_component(cache_key: str, component: nn.Module) -> None:
    """Store a component in cache."""
    _component_cache[cache_key] = weakref.ref(component)
    log.debug(f"Cached component with key: {cache_key}")


def generate_cache_key(component_type: str, config: dict[str, Any]) -> str:
    """Generate a unique cache key based on type and config."""
    key_parts = [component_type]
    for k, v in sorted(config.items()):
        # Skip 'type' key as it's already part of the key
        if k == "type":
            continue

        # Handle different value types consistently
        if isinstance(v, str | int | float | bool | type(None)):
            key_parts.append(f"{k}:{v}")
        elif isinstance(v, list | tuple):
            # Convert list/tuple elements to string
            key_parts.append(f"{k}:{','.join(map(str, cast(list[Any], v)))}")
        elif isinstance(v, dict):
            # Handle nested dictionaries recursively or flatten them
            v_dict = cast(dict[str, Any], v)
            nested_parts = [
                f"{str(nk)}:{str(nv)}" for nk, nv in sorted(v_dict.items())
            ]
            key_parts.append(f"{k}:{{{','.join(nested_parts)}}}")
        # Add handling for other types if necessary
        # else:
        #     log.warning(f"Unhandled type in cache key generation: {type(v)}")

    return ":".join(key_parts)
