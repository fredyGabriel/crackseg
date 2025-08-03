"""LRU cache for optimizing triplet access and reducing disk I/O.

This module implements a Least Recently Used (LRU) cache specifically
designed for ResultTriplet objects, with configurable capacity and
automatic cleanup of stale entries.

Phase 2: Observer pattern + LRU cache for reactive gallery updates.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TypeVar

from .core import ResultTriplet

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry[T]:
    """Cache entry with metadata for LRU management."""

    value: T
    access_time: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def touch(self) -> None:
        """Update access time and increment access count."""
        self.access_time = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0

    @property
    def fill_rate(self) -> float:
        """Calculate cache fill rate as percentage."""
        return (
            (self.current_size / self.max_size * 100.0)
            if self.max_size > 0
            else 0.0
        )


class LRUCache[T]:
    """Thread-safe LRU cache with configurable capacity and statistics.

    Features:
    - Thread-safe operations using RLock
    - Configurable maximum capacity
    - Automatic eviction of least recently used items
    - Performance statistics and monitoring
    - Optional TTL (time-to-live) for entries

    Example:
        >>> cache = LRUCache[ResultTriplet](capacity=100)
        >>> cache.put("triplet_001", triplet)
        >>> triplet = cache.get("triplet_001")
        >>> print(f"Hit rate: {cache.stats.hit_rate:.1f}%")
    """

    def __init__(
        self,
        capacity: int = 100,
        ttl_seconds: float | None = None,
        enable_stats: bool = True,
    ) -> None:
        """Initialize the LRU cache.

        Args:
            capacity: Maximum number of items to store
            ttl_seconds: Time-to-live for entries (None = no expiration)
            enable_stats: Whether to collect performance statistics
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._ttl_seconds = ttl_seconds
        self._enable_stats = enable_stats

        # Thread-safe storage using OrderedDict for LRU ordering
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = RLock()

        # Performance statistics
        self._stats = CacheStats(max_size=capacity) if enable_stats else None

    def get(self, key: str) -> T | None:
        """Get an item from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                if self._stats:
                    self._stats.misses += 1
                return None

            # Check TTL expiration
            if self._is_expired(entry):
                self._cache.pop(key)
                if self._stats:
                    self._stats.misses += 1
                    self._stats.current_size = len(self._cache)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            if self._stats:
                self._stats.hits += 1

            return entry.value

    def put(self, key: str, value: T, size_bytes: int = 0) -> None:
        """Put an item into the cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Optional size in bytes for monitoring
        """
        with self._lock:
            # Update existing entry
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.size_bytes = size_bytes
                entry.touch()
                self._cache.move_to_end(key)
                return

            # Add new entry
            entry = CacheEntry(value=value, size_bytes=size_bytes)
            self._cache[key] = entry

            # Evict if over capacity
            while len(self._cache) > self._capacity:
                self._evict_lru()

            if self._stats:
                self._stats.current_size = len(self._cache)

    def remove(self, key: str) -> bool:
        """Remove an item from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if item was found and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                if self._stats:
                    self._stats.current_size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            if self._stats:
                self._stats.current_size = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        if self._ttl_seconds is None:
            return 0

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]

            for key in expired_keys:
                self._cache.pop(key)

            if self._stats and expired_keys:
                self._stats.current_size = len(self._cache)

            return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item
            if self._stats:
                self._stats.evictions += 1

    def _is_expired(self, entry: CacheEntry[T]) -> bool:
        """Check if a cache entry has expired."""
        if self._ttl_seconds is None:
            return False
        return (time.time() - entry.access_time) > self._ttl_seconds

    @property
    def stats(self) -> CacheStats:
        """Get cache performance statistics."""
        if not self._stats:
            return CacheStats()

        with self._lock:
            # Update current size
            self._stats.current_size = len(self._cache)
            return self._stats

    @property
    def capacity(self) -> int:
        """Get cache capacity."""
        return self._capacity

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> list[str]:
        """Get all cache keys (ordered by recency)."""
        with self._lock:
            return list(self._cache.keys())

    def contains(self, key: str) -> bool:
        """Check if key exists in cache (without updating access time)."""
        with self._lock:
            return key in self._cache


class TripletCache:
    """Specialized LRU cache for ResultTriplet objects.

    Provides domain-specific functionality for caching crack segmentation
    prediction triplets with automatic size calculation and validation.

    Example:
        >>> cache = TripletCache(capacity=50, ttl_minutes=30)
        >>> cache.cache_triplet(triplet)
        >>> cached_triplet = cache.get_triplet(triplet.id)
        >>> print(f"Cache efficiency: {cache.get_efficiency():.1f}%")
    """

    def __init__(
        self,
        capacity: int = 100,
        ttl_minutes: float | None = None,
        validate_files: bool = True,
    ) -> None:
        """Initialize the triplet cache.

        Args:
            capacity: Maximum number of triplets to cache
            ttl_minutes: Time-to-live in minutes (None = no expiration)
            validate_files: Whether to validate file existence on retrieval
        """
        ttl_seconds = ttl_minutes * 60 if ttl_minutes else None
        self._cache = LRUCache[ResultTriplet](
            capacity=capacity, ttl_seconds=ttl_seconds, enable_stats=True
        )
        self._validate_files = validate_files

    def cache_triplet(self, triplet: ResultTriplet) -> None:
        """Cache a result triplet.

        Args:
            triplet: ResultTriplet to cache
        """
        # Calculate approximate size
        size_bytes = self._calculate_triplet_size(triplet)
        self._cache.put(triplet.id, triplet, size_bytes)

        logger.debug(f"Cached triplet {triplet.id} ({size_bytes} bytes)")

    def get_triplet(self, triplet_id: str) -> ResultTriplet | None:
        """Get a cached triplet by ID.

        Args:
            triplet_id: ID of the triplet to retrieve

        Returns:
            Cached triplet if found and valid, None otherwise
        """
        triplet = self._cache.get(triplet_id)

        if triplet is None:
            return None

        # Validate file existence if enabled
        if self._validate_files and not triplet.is_complete:
            logger.warning(
                f"Triplet {triplet_id} has missing files, removing from cache"
            )
            self._cache.remove(triplet_id)
            return None

        return triplet

    def remove_triplet(self, triplet_id: str) -> bool:
        """Remove a triplet from the cache.

        Args:
            triplet_id: ID of the triplet to remove

        Returns:
            True if triplet was found and removed
        """
        return self._cache.remove(triplet_id)

    def get_cached_ids(self) -> list[str]:
        """Get all cached triplet IDs ordered by recency."""
        return self._cache.keys()

    def cleanup_invalid(self) -> int:
        """Remove triplets with missing files from cache.

        Returns:
            Number of invalid triplets removed
        """
        if not self._validate_files:
            return 0

        invalid_ids = []
        for triplet_id in self._cache.keys():
            triplet = self._cache.get(triplet_id)
            if triplet and not triplet.is_complete:
                invalid_ids.append(triplet_id)

        for triplet_id in invalid_ids:
            self._cache.remove(triplet_id)

        if invalid_ids:
            logger.info(
                f"Removed {len(invalid_ids)} invalid triplets from cache"
            )

        return len(invalid_ids)

    def get_efficiency(self) -> float:
        """Get cache hit rate as percentage."""
        return self._cache.stats.hit_rate

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self._cache.stats
        return {
            "hit_rate": stats.hit_rate,
            "fill_rate": stats.fill_rate,
            "hits": stats.hits,
            "misses": stats.misses,
            "evictions": stats.evictions,
            "current_size": stats.current_size,
            "max_size": stats.max_size,
            "cached_triplets": len(self._cache.keys()),
        }

    def clear(self) -> None:
        """Clear all cached triplets."""
        self._cache.clear()
        logger.info("Cleared triplet cache")

    def _calculate_triplet_size(self, triplet: ResultTriplet) -> int:
        """Calculate approximate size of a triplet in bytes."""
        try:
            file_sizes = triplet.file_sizes
            return sum(file_sizes.values())
        except Exception as e:
            logger.warning(
                f"Could not calculate size for triplet {triplet.id}: {e}"
            )
            return 0


# Global triplet cache instance for convenience
_global_triplet_cache: TripletCache | None = None


def get_triplet_cache(
    capacity: int = 100, ttl_minutes: float | None = 30.0
) -> TripletCache:
    """Get the global triplet cache instance.

    Args:
        capacity: Cache capacity (only used on first call)
        ttl_minutes: TTL in minutes (only used on first call)

    Returns:
        Global TripletCache instance
    """
    global _global_triplet_cache
    if _global_triplet_cache is None:
        _global_triplet_cache = TripletCache(
            capacity=capacity, ttl_minutes=ttl_minutes
        )
    return _global_triplet_cache


def reset_triplet_cache() -> None:
    """Reset the global triplet cache (useful for testing)."""
    global _global_triplet_cache
    if _global_triplet_cache:
        _global_triplet_cache.clear()
    _global_triplet_cache = None
