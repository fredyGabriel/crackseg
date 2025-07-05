"""Parallel test execution configuration and optimization framework.

This module provides comprehensive configuration management for parallel test
execution using pytest-xdist, with support for different execution strategies,
resource management, and performance monitoring integration.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import psutil

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for parallel testing."""

    SEQUENTIAL = "sequential"  # Run tests one by one
    PARALLEL_BY_TEST = "parallel_test"  # Distribute individual tests
    PARALLEL_BY_FILE = "parallel_file"  # Distribute test files
    PARALLEL_BY_CLASS = "parallel_class"  # Distribute test classes
    DISTRIBUTED = "distributed"  # Use distributed execution
    BROWSER_MATRIX = "browser_matrix"  # Parallel across browsers


class WorkerLoadStrategy(Enum):
    """Load balancing strategies for workers."""

    ROUND_ROBIN = "loadscope"  # Distribute by scope (pytest-xdist default)
    WORK_STEALING = "worksteal"  # Work stealing algorithm
    GROUP_BY_FILE = "loadfile"  # Group tests by file
    GROUP_BY_CLASS = "loadgroup"  # Group tests by class/module


@dataclass
class ResourceLimits:
    """Resource limits for parallel test execution."""

    max_workers: int = 4
    max_memory_mb: int = 8192  # 8GB max memory usage
    max_cpu_percent: float = 80.0  # Max 80% CPU usage
    browser_pool_size: int = 6
    port_range_start: int = 8600
    port_range_end: int = 8699

    def validate(self) -> None:
        """Validate resource limits against system capabilities."""
        system_cores = psutil.cpu_count(logical=True) or 4
        system_memory_mb = psutil.virtual_memory().total // 1024 // 1024

        if self.max_workers > system_cores:
            logger.warning(
                f"max_workers ({self.max_workers}) exceeds system cores "
                f"({system_cores}), limiting to {system_cores}"
            )
            self.max_workers = min(self.max_workers, system_cores)

        if self.max_memory_mb > system_memory_mb * 0.8:
            recommended = int(system_memory_mb * 0.8)
            logger.warning(
                f"max_memory_mb ({self.max_memory_mb}) exceeds 80% of system "
                f"memory ({system_memory_mb}), limiting to {recommended}"
            )
            self.max_memory_mb = recommended


@dataclass
class ParallelTestConfig:
    """Comprehensive configuration for parallel test execution."""

    # Execution strategy
    strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_BY_TEST
    worker_load_strategy: WorkerLoadStrategy = WorkerLoadStrategy.WORK_STEALING

    # Resource management
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Performance monitoring
    performance_monitoring_enabled: bool = False
    performance_threshold_seconds: float = 30.0
    performance_regression_detection: bool = False

    # Test isolation
    isolation_level: Literal["process", "thread", "container"] = "process"
    cleanup_between_tests: bool = True
    shared_state_management: bool = True

    # Error handling
    fail_fast: bool = False
    max_test_retries: int = 2
    retry_delay_seconds: float = 1.0
    continue_on_collection_errors: bool = False

    # Reporting
    generate_worker_reports: bool = True
    merge_coverage_reports: bool = True
    generate_performance_summary: bool = True

    # Advanced settings
    pytest_xdist_args: list[str] = field(default_factory=list)
    custom_worker_timeout: int = 300  # 5 minutes per worker
    enable_load_balancing: bool = True

    def __post_init__(self) -> None:
        """Validate and optimize configuration after initialization."""
        self.resource_limits.validate()
        self._optimize_for_system()

    def _optimize_for_system(self) -> None:
        """Optimize configuration based on system characteristics."""
        # Auto-detect optimal worker count if not explicitly set
        if hasattr(self.resource_limits, "_auto_workers"):
            cores = psutil.cpu_count(logical=True) or 4
            memory_gb = psutil.virtual_memory().total // (1024**3)

            # For E2E tests: conservative approach due to browser overhead
            if memory_gb >= 16:
                optimal_workers = min(cores - 1, 6)  # Leave 1 core free
            elif memory_gb >= 8:
                optimal_workers = min(cores - 1, 4)
            else:
                optimal_workers = 2  # Minimal parallel for low memory

            self.resource_limits.max_workers = optimal_workers
            logger.info(
                f"Auto-optimized workers: {optimal_workers} (cores: {cores}, "
                f"memory: {memory_gb}GB)"
            )

    def to_pytest_args(self) -> list[str]:
        """Convert configuration to pytest-xdist command line arguments.

        Returns:
            List of pytest arguments for parallel execution
        """
        args = []

        # Worker count
        if self.strategy != ExecutionStrategy.SEQUENTIAL:
            if self.resource_limits.max_workers == -1:
                args.extend(["-n", "auto"])
            else:
                args.extend(["-n", str(self.resource_limits.max_workers)])

        # Load balancing strategy
        if self.enable_load_balancing:
            args.extend(["--dist", self.worker_load_strategy.value])

        # Timeout
        args.extend(["--timeout", str(self.custom_worker_timeout)])

        # Retry configuration
        if self.max_test_retries > 0:
            args.extend(["--reruns", str(self.max_test_retries)])
            args.extend(["--reruns-delay", str(self.retry_delay_seconds)])

        # Fail fast
        if self.fail_fast:
            args.append("-x")

        # Continue on collection errors
        if self.continue_on_collection_errors:
            args.append("--continue-on-collection-errors")

        # Performance markers
        if self.performance_monitoring_enabled:
            args.extend(
                [
                    "-m",
                    "performance or performance_critical or "
                    "performance_baseline",
                ]
            )

        # Custom pytest-xdist arguments
        args.extend(self.pytest_xdist_args)

        return args

    def get_worker_env_vars(self, worker_id: str) -> dict[str, str]:
        """Get environment variables for a specific worker.

        Args:
            worker_id: Unique identifier for the worker

        Returns:
            Dictionary of environment variables
        """
        worker_index = (
            int(worker_id.replace("gw", "")) if "gw" in worker_id else 0
        )

        # Assign unique ports to avoid conflicts
        base_port = self.resource_limits.port_range_start + (worker_index * 10)

        return {
            "PYTEST_WORKER_ID": worker_id,
            "PYTEST_WORKER_INDEX": str(worker_index),
            "STREAMLIT_SERVER_PORT": str(base_port),
            "SELENIUM_GRID_PORT": str(base_port + 1),
            "TEST_ARTIFACTS_PREFIX": f"worker_{worker_id}",
            "PARALLEL_EXECUTION": "true",
            "WORKER_ISOLATION_LEVEL": self.isolation_level,
        }


def create_parallel_config(
    strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_BY_TEST,
    max_workers: int | None = None,
    enable_performance: bool = False,
    **kwargs: Any,
) -> ParallelTestConfig:
    """Create optimized parallel test configuration.

    Args:
        strategy: Execution strategy to use
        max_workers: Maximum number of workers (auto-detect if None)
        enable_performance: Enable performance monitoring
        **kwargs: Additional configuration parameters

    Returns:
        Configured ParallelTestConfig instance
    """
    # Auto-detect optimal worker count
    if max_workers is None:
        cores = psutil.cpu_count(logical=True) or 4
        memory_gb = psutil.virtual_memory().total // (1024**3)

        if memory_gb >= 16:
            max_workers = min(cores - 1, 6)
        elif memory_gb >= 8:
            max_workers = min(cores - 1, 4)
        else:
            max_workers = 2

    resource_limits = ResourceLimits(max_workers=max_workers)

    config = ParallelTestConfig(
        strategy=strategy,
        resource_limits=resource_limits,
        performance_monitoring_enabled=enable_performance,
        **kwargs,
    )

    logger.info(
        f"Created parallel config: strategy={strategy.value}, "
        f"workers={max_workers}, performance={enable_performance}"
    )

    return config


def detect_optimal_strategy() -> ExecutionStrategy:
    """Detect optimal execution strategy based on system characteristics.

    Returns:
        Recommended execution strategy
    """
    cores = psutil.cpu_count(logical=True) or 4
    memory_gb = psutil.virtual_memory().total // (1024**3)

    # For CI environments (detected by env vars)
    if any(
        env in os.environ for env in ["CI", "GITHUB_ACTIONS", "JENKINS_URL"]
    ):
        return ExecutionStrategy.PARALLEL_BY_FILE  # More stable for CI

    # For high-resource systems
    if cores >= 8 and memory_gb >= 16:
        return ExecutionStrategy.PARALLEL_BY_TEST  # Maximum parallelization

    # For medium-resource systems
    if cores >= 4 and memory_gb >= 8:
        return ExecutionStrategy.PARALLEL_BY_FILE  # Balanced approach

    # For low-resource systems
    return ExecutionStrategy.SEQUENTIAL  # Conservative approach


# Predefined configurations for common scenarios
CONFIGS = {
    "dev": create_parallel_config(
        strategy=ExecutionStrategy.PARALLEL_BY_TEST,
        max_workers=4,
        enable_performance=True,
        fail_fast=True,
    ),
    "ci": create_parallel_config(
        strategy=ExecutionStrategy.PARALLEL_BY_FILE,
        max_workers=3,
        enable_performance=False,
        continue_on_collection_errors=True,
    ),
    "performance": create_parallel_config(
        strategy=ExecutionStrategy.SEQUENTIAL,
        max_workers=1,
        enable_performance=True,
        performance_regression_detection=True,
    ),
    "smoke": create_parallel_config(
        strategy=ExecutionStrategy.PARALLEL_BY_FILE,
        max_workers=2,
        enable_performance=False,
        fail_fast=True,
    ),
    "full": create_parallel_config(
        strategy=ExecutionStrategy.BROWSER_MATRIX,
        max_workers=6,
        enable_performance=True,
        max_test_retries=3,
    ),
}


def get_predefined_config(config_name: str) -> ParallelTestConfig:
    """Get a predefined parallel test configuration by name.

    Args:
        config_name: Name of the predefined configuration
            ('dev', 'ci', 'smoke', 'performance', 'full')

    Returns:
        ParallelTestConfig instance for the specified configuration

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(
            f"Unknown config name '{config_name}'. Available: {available}"
        )

    return CONFIGS[config_name]
