"""Test Environment Manager for CrackSeg E2E Testing.

This module provides centralized test environment setup with consistent
hardware specifications, environment isolation, and baseline network
conditions. Designed for Subtask 16.1 - Test Environment Setup.
"""

import logging
import platform
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from docker.env_manager import EnvironmentConfig, EnvironmentManager

from ..config.browser_config_manager import BrowserConfigManager, BrowserMatrix
from ..config.resource_manager import ResourceAllocation, ResourceManager

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpecification:
    """Consistent hardware specifications for test environment."""

    cpu_cores: int = 2
    memory_mb: int = 4096
    disk_space_mb: int = 10240
    network_bandwidth_mbps: int = 100
    browser_instances: int = 3
    concurrent_tests: int = 4

    # RTX 3070 Ti specific constraints
    gpu_memory_mb: int = 8192
    max_model_batch_size: int = 16

    def validate(self) -> None:
        """Validate hardware specifications."""
        if self.cpu_cores < 1:
            raise ValueError(f"CPU cores must be >= 1, got {self.cpu_cores}")
        if self.memory_mb < 1024:
            raise ValueError(f"Memory must be >= 1024MB, got {self.memory_mb}")
        if self.concurrent_tests < 1:
            raise ValueError(
                f"Concurrent tests must be >= 1, got {self.concurrent_tests}"
            )


@dataclass
class NetworkConditions:
    """Baseline network conditions for consistent testing."""

    latency_ms: int = 50
    bandwidth_mbps: int = 100
    packet_loss_percent: float = 0.0
    jitter_ms: int = 10
    connection_timeout_sec: int = 30
    retry_attempts: int = 3

    def validate(self) -> None:
        """Validate network conditions."""
        if self.latency_ms < 0:
            raise ValueError(f"Latency must be >= 0, got {self.latency_ms}")
        if self.bandwidth_mbps <= 0:
            raise ValueError(
                f"Bandwidth must be > 0, got {self.bandwidth_mbps}"
            )
        if not (0.0 <= self.packet_loss_percent <= 100.0):
            raise ValueError(
                f"Packet loss must be 0-100%, got {self.packet_loss_percent}"
            )


@dataclass
class EnvironmentIsolation:
    """Environment isolation configuration."""

    process_isolation: bool = True
    network_isolation: bool = True
    filesystem_isolation: bool = True
    port_range: tuple[int, int] = (8600, 8699)
    temp_dir_prefix: str = "crackseg_test_"
    cleanup_on_exit: bool = True

    # Resource limits
    memory_limit_mb: int = 4096
    cpu_limit_percent: int = 80
    disk_limit_mb: int = 10240

    def validate(self) -> None:
        """Validate isolation configuration."""
        start_port, end_port = self.port_range
        if start_port >= end_port:
            raise ValueError(f"Invalid port range: {self.port_range}")
        if not (1024 <= start_port <= 65535):
            raise ValueError(f"Invalid start port: {start_port}")
        if not (1024 <= end_port <= 65535):
            raise ValueError(f"Invalid end port: {end_port}")


@dataclass
class TestEnvironmentConfig:
    """Complete test environment configuration."""

    hardware: HardwareSpecification = field(
        default_factory=HardwareSpecification
    )
    network: NetworkConditions = field(default_factory=NetworkConditions)
    isolation: EnvironmentIsolation = field(
        default_factory=EnvironmentIsolation
    )

    # Environment identification
    environment_id: str = ""
    test_session_id: str = ""

    # Paths
    artifacts_dir: Path = Path("test-artifacts")
    temp_dir: Path = Path("test-temp")

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.hardware.validate()
        self.network.validate()
        self.isolation.validate()

        # Generate IDs if not provided
        if not self.environment_id:
            self.environment_id = f"crackseg_e2e_{int(time.time())}"
        if not self.test_session_id:
            self.test_session_id = f"session_{int(time.time())}"


class SystemCapabilityChecker:
    """Checks system capabilities against requirements."""

    def __init__(self) -> None:
        """Initialize system capability checker."""
        self.logger = logging.getLogger(f"{__name__}.SystemCapabilityChecker")

    def check_hardware_compatibility(
        self, spec: HardwareSpecification
    ) -> dict[str, Any]:
        """Check if system meets hardware requirements.

        Args:
            spec: Hardware specification to check against

        Returns:
            Dictionary with compatibility check results
        """
        results: dict[str, Any] = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "system_info": {},
        }

        # Check CPU cores
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None:
            cpu_count = 1  # Fallback to 1 if unable to determine
            results["warnings"].append(
                "Unable to determine CPU count, using fallback"
            )

        results["system_info"]["cpu_cores"] = cpu_count
        if cpu_count < spec.cpu_cores:
            results["warnings"].append(
                f"System has {cpu_count} CPU cores, requires {spec.cpu_cores}"
            )

        # Check memory
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.total / (1024 * 1024)
        results["system_info"]["memory_mb"] = int(memory_mb)
        if memory_mb < spec.memory_mb:
            results["errors"].append(
                f"System has {memory_mb:.0f}MB memory, requires "
                f"{spec.memory_mb}MB"
            )
            results["compatible"] = False

        # Check disk space
        disk_info = psutil.disk_usage(".")
        disk_mb = disk_info.free / (1024 * 1024)
        results["system_info"]["disk_free_mb"] = int(disk_mb)
        if disk_mb < spec.disk_space_mb:
            results["warnings"].append(
                f"System has {disk_mb:.0f}MB free disk, requires "
                f"{spec.disk_space_mb}MB"
            )

        # Check platform
        results["system_info"]["platform"] = platform.system()
        results["system_info"]["platform_version"] = platform.version()

        return results

    def check_network_conditions(
        self, conditions: NetworkConditions
    ) -> dict[str, Any]:
        """Check network conditions.

        Args:
            conditions: Network conditions to validate

        Returns:
            Dictionary with network check results
        """
        results: dict[str, Any] = {
            "reachable": True,
            "latency_ok": True,
            "bandwidth_ok": True,
            "warnings": [],
        }

        # Basic network reachability check
        try:
            import socket

            socket.create_connection(
                ("8.8.8.8", 53), timeout=conditions.connection_timeout_sec
            )
        except Exception as e:
            results["reachable"] = False
            results["warnings"].append(f"Network unreachable: {e}")

        self.logger.info(f"Network conditions check: {results}")
        return results


class TestEnvironmentManager:
    """Manages test environment setup with consistent specifications."""

    def __init__(
        self,
        config: TestEnvironmentConfig | None = None,
        resource_manager: ResourceManager | None = None,
        browser_config_manager: BrowserConfigManager | None = None,
        env_manager: EnvironmentManager | None = None,
    ) -> None:
        """Initialize test environment manager.

        Args:
            config: Test environment configuration
            resource_manager: Resource manager for isolation
            browser_config_manager: Browser configuration manager
            env_manager: Environment variable manager
        """
        self.config = config or TestEnvironmentConfig()
        self.resource_manager = resource_manager or ResourceManager()
        self.browser_config_manager = (
            browser_config_manager or BrowserConfigManager()
        )
        self.env_manager = env_manager or EnvironmentManager()

        self.capability_checker = SystemCapabilityChecker()
        self.logger = logging.getLogger(f"{__name__}.TestEnvironmentManager")

        # State tracking
        self._active_allocations: dict[str, ResourceAllocation] = {}
        self._environment_setup: bool = False
        self._temp_dirs: list[Path] = []

    def validate_environment(self) -> dict[str, Any]:
        """Validate entire test environment setup.

        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating test environment setup...")

        validation_results: dict[str, Any] = {
            "valid": True,
            "hardware": {},
            "network": {},
            "isolation": {},
            "warnings": [],
            "errors": [],
        }

        # Check hardware compatibility
        hardware_results = (
            self.capability_checker.check_hardware_compatibility(
                self.config.hardware
            )
        )
        validation_results["hardware"] = hardware_results

        if not hardware_results["compatible"]:
            validation_results["valid"] = False
            validation_results["errors"].extend(hardware_results["errors"])

        validation_results["warnings"].extend(hardware_results["warnings"])

        # Check network conditions
        network_results = self.capability_checker.check_network_conditions(
            self.config.network
        )
        validation_results["network"] = network_results

        if not network_results["reachable"]:
            validation_results["warnings"].append(
                "Network connectivity issues detected"
            )

        # Validate isolation settings
        try:
            self.config.isolation.validate()
            validation_results["isolation"]["valid"] = True
        except ValueError as e:
            validation_results["isolation"]["valid"] = False
            validation_results["errors"].append(
                f"Isolation validation failed: {e}"
            )
            validation_results["valid"] = False

        self.logger.info(
            f"Environment validation completed: {validation_results['valid']}"
        )
        return validation_results

    @contextmanager
    def setup_test_environment(self) -> Generator[dict[str, Any], None, None]:
        """Context manager for complete test environment setup.

        Yields:
            Dictionary with environment setup information
        """
        self.logger.info(
            f"Setting up test environment: {self.config.environment_id}"
        )

        # Validate environment first
        validation = self.validate_environment()
        if not validation["valid"]:
            raise RuntimeError(
                f"Environment validation failed: {validation['errors']}"
            )

        # Setup environment variables
        env_config = self._create_env_config()
        self.env_manager.apply_configuration(env_config)

        # Create directories
        self._setup_directories()

        # Acquire resources with isolation
        with self.resource_manager.acquire_resources(
            memory_limit_mb=self.config.isolation.memory_limit_mb,
            cpu_limit=self.config.isolation.cpu_limit_percent,
            port_count=3,
            worker_id=self.config.environment_id,
        ) as resource_allocation:
            self._active_allocations[self.config.environment_id] = (
                resource_allocation
            )

            # Setup browser configuration
            browser_matrix = self._create_browser_matrix()

            environment_info = {
                "environment_id": self.config.environment_id,
                "test_session_id": self.config.test_session_id,
                "resource_allocation": resource_allocation,
                "browser_matrix": browser_matrix,
                "validation_results": validation,
                "directories": {
                    "artifacts": str(self.config.artifacts_dir),
                    "temp": str(self.config.temp_dir),
                },
                "network_conditions": self.config.network,
                "hardware_specs": self.config.hardware,
            }

            self._environment_setup = True
            self.logger.info(
                "Test environment setup completed: "
                f"{self.config.environment_id}"
            )

            try:
                yield environment_info
            finally:
                self._cleanup_environment()

    def _create_env_config(self) -> EnvironmentConfig:
        """Create environment configuration for testing."""
        return EnvironmentConfig(
            node_env="test",
            crackseg_env="test",
            test_headless=True,
            test_parallel_workers=self.config.hardware.concurrent_tests,
            test_timeout=self.config.network.connection_timeout_sec,
            max_browser_instances=self.config.hardware.browser_instances,
            memory_limit=f"{self.config.isolation.memory_limit_mb}m",
            cpu_limit=str(self.config.isolation.cpu_limit_percent),
            test_artifacts_path=str(self.config.artifacts_dir),
            selenium_implicit_wait=self.config.network.latency_ms // 10,
            selenium_page_load_timeout=self.config.network.connection_timeout_sec,
        )

    def _create_browser_matrix(self) -> BrowserMatrix:
        """Create browser matrix for testing."""
        return BrowserMatrix(
            browsers=["chrome", "firefox"],
            headless_modes=[True],
            window_sizes=[(1920, 1080), (1366, 768)],
            include_mobile=False,
        )

    def _setup_directories(self) -> None:
        """Setup required directories for testing."""
        directories = [
            self.config.artifacts_dir,
            self.config.temp_dir,
            self.config.artifacts_dir / "screenshots",
            self.config.artifacts_dir / "videos",
            self.config.artifacts_dir / "logs",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.name.startswith("test-temp"):
                self._temp_dirs.append(directory)

        self.logger.debug(f"Created {len(directories)} directories")

    def _cleanup_environment(self) -> None:
        """Clean up environment resources."""
        if not self.config.isolation.cleanup_on_exit:
            return

        self.logger.info("Cleaning up test environment...")

        # Clean up temporary directories
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    import shutil

                    shutil.rmtree(temp_dir)
                    self.logger.debug(
                        f"Removed temporary directory: {temp_dir}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to remove {temp_dir}: {e}")

        # Clear active allocations
        self._active_allocations.clear()
        self._environment_setup = False

        self.logger.info("Test environment cleanup completed")

    def get_environment_status(self) -> dict[str, Any]:
        """Get current environment status.

        Returns:
            Dictionary with environment status information
        """
        return {
            "environment_id": self.config.environment_id,
            "setup_complete": self._environment_setup,
            "active_allocations": len(self._active_allocations),
            "temp_directories": len(self._temp_dirs),
            "config": {
                "hardware": self.config.hardware,
                "network": self.config.network,
                "isolation": self.config.isolation,
            },
        }

    def reset_environment(self) -> None:
        """Reset environment to clean state."""
        self.logger.info("Resetting test environment...")

        self._cleanup_environment()
        self._temp_dirs.clear()

        # Generate new IDs
        self.config.environment_id = f"crackseg_e2e_{int(time.time())}"
        self.config.test_session_id = f"session_{int(time.time())}"

        self.logger.info("Test environment reset completed")
