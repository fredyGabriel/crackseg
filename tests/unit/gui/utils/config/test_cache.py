"""Unit tests for the configuration cache."""

import time
from pathlib import Path

import pytest

from scripts.gui.utils.config.cache import ConfigCache


@pytest.fixture
def cache() -> ConfigCache:
    """Fixture to provide a clean ConfigCache instance for each test."""
    return ConfigCache(maxsize=3)


class TestConfigCache:
    """Test suite for the ConfigCache class."""

    def test_set_and_get_item(self, cache: ConfigCache, tmp_path: Path):
        """Test setting and getting an item from the cache."""
        file_path = tmp_path / "config.yaml"
        file_path.touch()
        config_data: dict[str, object] = {"key": "value"}

        cache.set(str(file_path), config_data)
        retrieved = cache.get(str(file_path))

        assert retrieved == config_data

    def test_get_non_existent_item(self, cache: ConfigCache):
        """Test getting a non-existent item returns None."""
        assert cache.get("/non/existent/path.yaml") is None

    def test_invalidation_on_file_modification(
        self, cache: ConfigCache, tmp_path: Path
    ):
        """Test that the cache is invalidated if the file is modified."""
        file_path = tmp_path / "config.yaml"
        file_path.write_text("initial")
        config_data: dict[str, object] = {"key": "initial_value"}

        # Set initial item in cache
        cache.set(str(file_path), config_data)
        assert cache.get(str(file_path)) == config_data

        # Modify the file
        time.sleep(0.1)  # Ensure modification time is different
        file_path.write_text("updated")

        # Now, getting the item should return None (cache invalidated)
        assert cache.get(str(file_path)) is None
        assert str(file_path) not in cache._cache

    def test_invalidation_on_file_deletion(
        self, cache: ConfigCache, tmp_path: Path
    ):
        """Test that the cache is invalidated if the file is deleted."""
        file_path = tmp_path / "config.yaml"
        file_path.touch()
        config_data: dict[str, object] = {"key": "value"}

        cache.set(str(file_path), config_data)
        assert cache.get(str(file_path)) is not None

        # Delete the file
        file_path.unlink()

        # Getting the item should now return None
        assert cache.get(str(file_path)) is None

    def test_lru_eviction_policy(self, cache: ConfigCache, tmp_path: Path):
        """
        Test that the least recently used item is evicted when maxsize is
        reached.
        """
        # cache maxsize is 3
        paths = [tmp_path / f"c{i}.yaml" for i in range(4)]
        for p in paths:
            p.touch()

        # Fill the cache
        config0: dict[str, object] = {"id": 0}
        config1: dict[str, object] = {"id": 1}
        config2: dict[str, object] = {"id": 2}
        cache.set(str(paths[0]), config0)
        cache.set(str(paths[1]), config1)
        cache.set(str(paths[2]), config2)

        # Access path 0 to make it recently used
        cache.get(str(paths[0]))

        # Add a new item, which should evict the LRU item (path 1)
        config3: dict[str, object] = {"id": 3}
        cache.set(str(paths[3]), config3)

        assert str(paths[0]) in cache._cache
        assert str(paths[1]) not in cache._cache  # Evicted
        assert str(paths[2]) in cache._cache
        assert str(paths[3]) in cache._cache

    def test_clear_cache(self, cache: ConfigCache, tmp_path: Path):
        """Test that the clear method removes all items."""
        file_path = tmp_path / "config.yaml"
        file_path.touch()
        config_data: dict[str, object] = {"key": "value"}
        cache.set(str(file_path), config_data)

        assert len(cache._cache) == 1
        cache.clear()
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
