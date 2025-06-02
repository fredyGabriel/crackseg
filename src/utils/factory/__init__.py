"""Factory and component creation utilities for the Crack Segmentation project.

This module provides factory patterns for creating components and component
caching.
"""

from .cache import (
    cache_component,
    clear_component_cache,
    generate_cache_key,
    get_cached_component,
)
from .factory import (
    get_loss_fn,
    get_metrics_from_cfg,
    get_optimizer,
    import_class,
)

__all__ = [
    # Component cache
    "cache_component",
    "clear_component_cache",
    "generate_cache_key",
    "get_cached_component",
    # Factory functions
    "get_loss_fn",
    "get_metrics_from_cfg",
    "get_optimizer",
    "import_class",
]
