"""
Test performance optimization framework. This module provides
comprehensive test performance optimization including fixture caching,
selective test running, and parallel execution strategies. Part of
subtask 7.5 - Test Execution Performance Optimization.
"""

import hashlib
import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for test execution."""

    test_name: str
    execution_time: float
    setup_time: float
    fixture_cache_hits: int = 0
    fixture_cache_misses: int = 0
    parallel_efficiency: float = 1.0
    memory_usage_mb: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for test performance optimization."""

    # Fixture optimization
    enable_fixture_caching: bool = True
    fixture_cache_ttl_seconds: int = 3600  # 1 hour
    max_fixture_cache_size_mb: int = 500

    # Selective test running
    enable_selective_running: bool = True
    dependency_tracking: bool = True
    changed_files_threshold: int = 50

    # Parallel execution
    auto_parallel_detection: bool = True
    optimal_worker_count: int = -1  # Auto-detect
    load_balancing_strategy: str = "worksteal"

    # Performance monitoring
    performance_baseline_enabled: bool = True
    regression_threshold_percent: float = 15.0
    metrics_collection_enabled: bool = True


class FixtureCache:
    """Intelligent caching system for expensive test fixtures."""

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize fixture cache.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.cache_dir = Path("test-artifacts") / "fixture-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_metadata: dict[str, dict[str, Any]] = {}
        self._load_cache_metadata()

    def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            try:
                with metadata_file.open() as f:
                    self._cache_metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to load cache metadata, starting fresh")
                self._cache_metadata = {}

    def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        try:
            with metadata_file.open("w") as f:
                json.dump(self._cache_metadata, f)
        except OSError:
            logger.warning("Failed to save cache metadata")

    def _get_cache_key(self, fixture_name: str, args: tuple[Any, ...]) -> str:
        """Generate cache key for fixture.

        Args:
            fixture_name: Name of the fixture
            args: Fixture arguments

        Returns:
            Cache key string
        """
        content = f"{fixture_name}:{str(args)}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_fixture(
        self, fixture_name: str, args: tuple[Any, ...]
    ) -> tuple[bool, Any]:
        """Retrieve cached fixture if available and valid.

        Args:
            fixture_name: Name of the fixture
            args: Fixture arguments

        Returns:
            Tuple of (cache_hit, cached_value)
        """
        if not self.config.enable_fixture_caching:
            return False, None

        cache_key = self._get_cache_key(fixture_name, args)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return False, None

        # Check TTL
        metadata = self._cache_metadata.get(cache_key, {})
        cache_time = metadata.get("timestamp", 0)
        if time.time() - cache_time > self.config.fixture_cache_ttl_seconds:
            cache_file.unlink(missing_ok=True)
            return False, None

        try:
            with cache_file.open("rb") as f:
                cached_value = pickle.load(f)
            logger.debug(f"Cache hit for fixture {fixture_name}")
            return True, cached_value
        except (OSError, pickle.PickleError):
            logger.warning(f"Failed to load cached fixture {fixture_name}")
            cache_file.unlink(missing_ok=True)
            return False, None

    def cache_fixture(
        self, fixture_name: str, args: tuple[Any, ...], value: Any
    ) -> None:
        """Cache fixture value.

        Args:
            fixture_name: Name of the fixture
            args: Fixture arguments
            value: Fixture value to cache
        """
        if not self.config.enable_fixture_caching:
            return

        cache_key = self._get_cache_key(fixture_name, args)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with cache_file.open("wb") as f:
                pickle.dump(value, f)

            # Update metadata
            self._cache_metadata[cache_key] = {
                "fixture_name": fixture_name,
                "timestamp": time.time(),
                "size_bytes": cache_file.stat().st_size,
            }
            self._save_cache_metadata()
            logger.debug(f"Cached fixture {fixture_name}")
        except (OSError, pickle.PickleError):
            logger.warning(f"Failed to cache fixture {fixture_name}")


class SelectiveTestRunner:
    """Selective test execution based on changes and dependencies."""

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize selective test runner.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self._dependency_map: dict[str, set[str]] = {}

    def should_run_test(
        self, test_path: str, changed_files: list[str]
    ) -> bool:
        """Determine if test should run based on changes.

        Args:
            test_path: Path to test file
            changed_files: List of changed files

        Returns:
            True if test should run
        """
        if not self.config.enable_selective_running:
            return True  # Always run if not enabled

        # Always run if too many files changed (likely major refactor)
        if len(changed_files) > self.config.changed_files_threshold:
            return True

        # Check direct dependencies
        test_dependencies = self._get_test_dependencies(test_path)
        if any(
            changed_file in test_dependencies for changed_file in changed_files
        ):
            return True

        return False

    def _get_test_dependencies(self, test_path: str) -> set[str]:
        """Get dependencies for a test file.

        Args:
            test_path: Path to test file

        Returns:
            Set of dependency file paths
        """
        # This would be enhanced with actual dependency analysis
        # For now, return basic heuristics
        dependencies = {test_path}

        # Add corresponding source file
        if "test_" in test_path:
            source_path = test_path.replace("tests/", "src/").replace(
                "test_", ""
            )
            dependencies.add(source_path)

        return dependencies


class ParallelOptimizer:
    """Optimize parallel test execution."""

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize parallel optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config

    def get_optimal_worker_count(self) -> int:
        """Determine optimal worker count for parallel execution.

        Returns:
            Optimal number of workers
        """
        if not self.config.auto_parallel_detection:
            return self.config.optimal_worker_count

        try:
            import psutil

            cpu_cores = psutil.cpu_count(logical=True) or 4
            memory_gb = psutil.virtual_memory().total // (1024**3)

            # Conservative optimization for test execution
            if memory_gb >= 16:
                return min(cpu_cores - 1, 6)
            elif memory_gb >= 8:
                return min(cpu_cores - 1, 4)
            else:
                return 2
        except ImportError:
            # Fallback if psutil not available
            return 2

    def get_pytest_args(self) -> list[str]:
        """Get optimized pytest arguments for parallel execution.

        Returns:
            List of pytest arguments
        """
        args = []

        # Worker count
        worker_count = self.get_optimal_worker_count()
        if worker_count > 1:
            args.extend(["-n", str(worker_count)])

        # Load balancing
        if self.config.load_balancing_strategy:
            args.extend(["--dist", self.config.load_balancing_strategy])

        return args


class TestPerformanceOptimizer:
    """Main test performance optimization coordinator."""

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        """Initialize performance optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.fixture_cache = FixtureCache(self.config)
        self.selective_runner = SelectiveTestRunner(self.config)
        self.parallel_optimizer = ParallelOptimizer(self.config)
        self.metrics: list[PerformanceMetrics] = []

    def get_optimized_pytest_command(
        self, base_args: list[str] | None = None
    ) -> list[str]:
        """Generate optimized pytest command.

        Args:
            base_args: Base pytest arguments

        Returns:
            Optimized pytest command
        """
        args = base_args or []

        # Add parallel optimization
        parallel_args = self.parallel_optimizer.get_pytest_args()
        args.extend(parallel_args)

        # Add performance monitoring
        if self.config.metrics_collection_enabled:
            args.extend(["--tb=short", "--durations=10"])

        return args

    def create_cached_fixture(
        self,
        fixture_func: Callable[..., Any],
        scope: Literal[
            "session", "package", "module", "class", "function"
        ] = "session",
    ) -> Callable[..., Any]:
        """Create a cached version of a fixture.

        Args:
            fixture_func: Original fixture function
            scope: Fixture scope

        Returns:
            Cached fixture function
        """

        def cached_fixture(*args: Any, **kwargs: Any) -> Any:
            fixture_name = fixture_func.__name__
            cache_args: tuple[Any, ...] = (args, tuple(sorted(kwargs.items())))

            # Try cache first
            cache_hit, cached_value = self.fixture_cache.get_cached_fixture(
                fixture_name, cache_args
            )

            if cache_hit:
                return cached_value

            # Compute and cache
            start_time = time.time()
            value = fixture_func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Cache if expensive (>1 second)
            if execution_time > 1.0:
                self.fixture_cache.cache_fixture(
                    fixture_name, cache_args, value
                )

            return value

        # Preserve pytest fixture metadata
        if hasattr(fixture_func, "_pytestfixturefunction"):
            cached_fixture._pytestfixturefunction = (
                fixture_func._pytestfixturefunction
            )

        return pytest.fixture(scope=scope)(cached_fixture)


# Singleton instance for easy access
_optimizer = TestPerformanceOptimizer()


def get_performance_optimizer() -> TestPerformanceOptimizer:
    """Get the global performance optimizer instance.

    Returns:
        TestPerformanceOptimizer instance
    """
    return _optimizer


def cached_fixture(
    scope: Literal[
        "session", "package", "module", "class", "function"
    ] = "session",
) -> Callable[[Callable[..., Any]], Any]:
    """Decorator for creating cached fixtures.

    Args:
        scope: Fixture scope

    Returns:
        Fixture decorator
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return _optimizer.create_cached_fixture(func, scope)

    return decorator
