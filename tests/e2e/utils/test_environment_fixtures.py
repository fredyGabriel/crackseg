"""Pytest fixtures for TestEnvironmentManager integration.

This module provides pytest fixtures that integrate the TestEnvironmentManager
with the existing E2E testing infrastructure, enabling consistent test
environment setup across all E2E tests.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from .test_environment_manager import (
    EnvironmentIsolation,
    HardwareSpecification,
    NetworkConditions,
    TestEnvironmentConfig,
    TestEnvironmentManager,
)


@pytest.fixture(scope="session")
def hardware_spec_default() -> HardwareSpecification:
    """Provide default hardware specification for E2E testing.

    Optimized for RTX 3070 Ti with 8GB VRAM constraints.

    Returns:
        HardwareSpecification: Default hardware specification
    """
    return HardwareSpecification(
        cpu_cores=2,
        memory_mb=4096,
        disk_space_mb=10240,
        network_bandwidth_mbps=100,
        browser_instances=2,  # Conservative for parallel testing
        concurrent_tests=3,
        gpu_memory_mb=8192,  # RTX 3070 Ti
        max_model_batch_size=8,  # Conservative for 8GB VRAM
    )


@pytest.fixture(scope="session")
def hardware_spec_high_performance() -> HardwareSpecification:
    """Provide high-performance hardware specification.

    For performance benchmarking tests that need maximum resources.

    Returns:
        HardwareSpecification: High-performance hardware specification
    """
    return HardwareSpecification(
        cpu_cores=4,
        memory_mb=8192,
        disk_space_mb=20480,
        network_bandwidth_mbps=1000,
        browser_instances=4,
        concurrent_tests=6,
        gpu_memory_mb=8192,
        max_model_batch_size=16,  # Maximum for RTX 3070 Ti
    )


@pytest.fixture(scope="session")
def network_conditions_default() -> NetworkConditions:
    """Provide default network conditions for E2E testing.

    Returns:
        NetworkConditions: Default network conditions
    """
    return NetworkConditions(
        latency_ms=50,
        bandwidth_mbps=100,
        packet_loss_percent=0.0,
        jitter_ms=10,
        connection_timeout_sec=30,
        retry_attempts=3,
    )


@pytest.fixture(scope="session")
def network_conditions_slow() -> NetworkConditions:
    """Provide slow network conditions for testing edge cases.

    Returns:
        NetworkConditions: Slow network conditions
    """
    return NetworkConditions(
        latency_ms=200,
        bandwidth_mbps=10,
        packet_loss_percent=1.0,
        jitter_ms=50,
        connection_timeout_sec=60,
        retry_attempts=5,
    )


@pytest.fixture(scope="session")
def environment_isolation_default() -> EnvironmentIsolation:
    """Provide default environment isolation configuration.

    Returns:
        EnvironmentIsolation: Default isolation configuration
    """
    return EnvironmentIsolation(
        process_isolation=True,
        network_isolation=True,
        filesystem_isolation=True,
        port_range=(8600, 8699),
        temp_dir_prefix="crackseg_test_",
        cleanup_on_exit=True,
        memory_limit_mb=4096,
        cpu_limit_percent=80,
        disk_limit_mb=10240,
    )


@pytest.fixture(scope="session")
def environment_isolation_strict() -> EnvironmentIsolation:
    """Provide strict environment isolation configuration.

    For tests that require maximum isolation and resource control.

    Returns:
        EnvironmentIsolation: Strict isolation configuration
    """
    return EnvironmentIsolation(
        process_isolation=True,
        network_isolation=True,
        filesystem_isolation=True,
        port_range=(8700, 8799),
        temp_dir_prefix="crackseg_strict_test_",
        cleanup_on_exit=True,
        memory_limit_mb=2048,  # Lower limit for strict isolation
        cpu_limit_percent=60,
        disk_limit_mb=5120,
    )


@pytest.fixture(scope="function")
def test_environment_config(
    hardware_spec_default: HardwareSpecification,
    network_conditions_default: NetworkConditions,
    environment_isolation_default: EnvironmentIsolation,
    test_artifacts_dir: Path,
) -> TestEnvironmentConfig:
    """Provide test environment configuration for function-scoped tests.

    Args:
        hardware_spec_default: Default hardware specification
        network_conditions_default: Default network conditions
        environment_isolation_default: Default isolation configuration
        test_artifacts_dir: Test artifacts directory from existing fixtures

    Returns:
        TestEnvironmentConfig: Complete test environment configuration
    """
    return TestEnvironmentConfig(
        hardware=hardware_spec_default,
        network=network_conditions_default,
        isolation=environment_isolation_default,
        artifacts_dir=test_artifacts_dir,
        temp_dir=test_artifacts_dir / "temp",
    )


@pytest.fixture(scope="function")
def test_environment_config_performance(
    hardware_spec_high_performance: HardwareSpecification,
    network_conditions_default: NetworkConditions,
    environment_isolation_default: EnvironmentIsolation,
    test_artifacts_dir: Path,
) -> TestEnvironmentConfig:
    """Provide performance-optimized test environment configuration.

    Args:
        hardware_spec_high_performance: High-performance hardware specification
        network_conditions_default: Default network conditions
        environment_isolation_default: Default isolation configuration
        test_artifacts_dir: Test artifacts directory from existing fixtures

    Returns:
        TestEnvironmentConfig: Performance-optimized test environment config
    """
    return TestEnvironmentConfig(
        hardware=hardware_spec_high_performance,
        network=network_conditions_default,
        isolation=environment_isolation_default,
        artifacts_dir=test_artifacts_dir / "performance",
        temp_dir=test_artifacts_dir / "performance" / "temp",
    )


@pytest.fixture(scope="function")
def test_environment_manager(
    test_environment_config: TestEnvironmentConfig,
) -> TestEnvironmentManager:
    """Provide TestEnvironmentManager instance for function-scoped tests.

    Args:
        test_environment_config: Test environment configuration

    Returns:
        TestEnvironmentManager: Configured test environment manager
    """
    return TestEnvironmentManager(config=test_environment_config)


@pytest.fixture(scope="function")
def test_environment_manager_performance(
    test_environment_config_performance: TestEnvironmentConfig,
) -> TestEnvironmentManager:
    """Provide performance-optimized TestEnvironmentManager instance.

    Args:
        test_environment_config_performance: Performance-optimized config

    Returns:
        TestEnvironmentManager: Performance-optimized test environment manager
    """
    return TestEnvironmentManager(config=test_environment_config_performance)


@pytest.fixture(scope="function")
def isolated_test_environment(
    test_environment_manager: TestEnvironmentManager,
) -> Generator[dict[str, Any], None, None]:
    """Provide isolated test environment with automatic setup and cleanup.

    This fixture provides a complete test environment with consistent hardware
    specifications, environment isolation, and baseline network conditions.

    Args:
        test_environment_manager: Test environment manager instance

    Yields:
        Dictionary with environment setup information
    """
    with test_environment_manager.setup_test_environment() as environment_info:
        yield environment_info


@pytest.fixture(scope="function")
def performance_test_environment(
    test_environment_manager_performance: TestEnvironmentManager,
) -> Generator[dict[str, Any], None, None]:
    """Provide performance-optimized test environment.

    This fixture provides a high-performance test environment specifically
    configured for performance benchmarking and resource-intensive tests.

    Args:
        test_environment_manager_performance: Performance-optimized manager

    Yields:
        Dictionary with environment setup information
    """
    manager = test_environment_manager_performance
    with manager.setup_test_environment() as environment_info:
        yield environment_info


@pytest.fixture(scope="function")
def environment_validation_results(
    test_environment_manager: TestEnvironmentManager,
) -> dict[str, Any]:
    """Provide environment validation results for tests.

    Args:
        test_environment_manager: Test environment manager instance

    Returns:
        Dictionary with validation results
    """
    return test_environment_manager.validate_environment()


# Pytest markers for test environment categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers for test environment management."""
    config.addinivalue_line(
        "markers",
        "isolated_environment: Tests that require isolated test environment",
    )
    config.addinivalue_line(
        "markers",
        "performance_environment: Tests requiring performance-optimized env",
    )
    config.addinivalue_line(
        "markers",
        "hardware_validation: Tests that validate hardware compatibility",
    )
    config.addinivalue_line(
        "markers", "network_validation: Tests that validate network conditions"
    )
