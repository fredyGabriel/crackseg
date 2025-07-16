"""LRU cache for configuration files with timestamp-based invalidation.

This module provides an LRU (Least Recently Used) cache for configuration files
that automatically invalidates cached entries when source files are modified.
"""

import os


class ConfigCache:
    """LRU cache for configuration files with timestamp-based invalidation."""

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize the configuration cache.

        Args:
            maxsize: Maximum number of cached configurations.
        """
        self._cache: dict[str, tuple[dict[str, object], float]] = {}
        self._maxsize = maxsize
        self._access_order: list[str] = []

    def get(self, path: str) -> dict[str, object] | None:
        """Get a configuration from cache if valid.

        Args:
            path: Path to the configuration file.

        Returns:
            Cached configuration dict or None if not cached/invalid.
        """
        if path not in self._cache:
            return None

        cached_config, cached_time = self._cache[path]

        # Check if file has been modified since caching
        try:
            current_mtime = os.path.getmtime(path)
            if current_mtime > cached_time:
                # File has been modified, invalidate cache
                self.invalidate(path)
                return None
        except OSError:
            # File might have been deleted
            self.invalidate(path)
            return None

        # Update access order
        if path in self._access_order:
            self._access_order.remove(path)
        self._access_order.append(path)

        return cached_config

    def set(self, path: str, config: dict[str, object]) -> None:
        """Cache a configuration.

        Args:
            path: Path to the configuration file.
            config: Configuration dictionary to cache.
        """
        # Enforce cache size limit
        if len(self._cache) >= self._maxsize and path not in self._cache:
            # Remove least recently used item
            if self._access_order:
                lru_path = self._access_order.pop(0)
                del self._cache[lru_path]

        # Cache with current modification time
        try:
            mtime = os.path.getmtime(path)
            self._cache[path] = (config, mtime)
            if path not in self._access_order:
                self._access_order.append(path)
        except OSError:
            # Don't cache if we can't get modification time
            pass

    def invalidate(self, path: str) -> None:
        """Invalidate a cached configuration.

        Args:
            path: Path to the configuration file to invalidate.
        """
        if path in self._cache:
            del self._cache[path]
            if path in self._access_order:
                self._access_order.remove(path)

    def clear(self) -> None:
        """Clear all cached configurations."""
        self._cache.clear()
        self._access_order.clear()


# Global cache instance for backward compatibility
_config_cache = ConfigCache()
