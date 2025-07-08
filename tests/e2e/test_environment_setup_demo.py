"""Demonstration tests for TestEnvironmentManager functionality.

This module provides comprehensive tests that validate the test environment
setup system and serve as usage examples for the TestEnvironmentManager.
Designed for Subtask 16.1 - Test Environment Setup validation.
"""

import logging
import time
from pathlib import Path
from typing import Any

import pytest

from .utils.test_environment_manager import (
    EnvironmentIsolation,
    HardwareSpecification,
    NetworkConditions,
    SystemCapabilityChecker,
    TestEnvironmentConfig,
    TestEnvironmentManager,
)

logger = logging.getLogger(__name__)


class TestEnvironmentSetupValidation:
    """Test suite for validating test environment setup functionality."""

    def test_hardware_specification_validation(self) -> None:
        """Test hardware specification validation logic."""
        # Valid specification
        valid_spec = HardwareSpecification(
            cpu_cores=2,
            memory_mb=4096,
            concurrent_tests=3,
        )
        valid_spec.validate()  # Should not raise

        # Invalid CPU cores
        with pytest.raises(ValueError, match="CPU cores must be >= 1"):
            invalid_spec = HardwareSpecification(cpu_cores=0)
            invalid_spec.validate()

        # Invalid memory
        with pytest.raises(ValueError, match="Memory must be >= 1024MB"):
            invalid_spec = HardwareSpecification(memory_mb=512)
            invalid_spec.validate()

        # Invalid concurrent tests
        with pytest.raises(ValueError, match="Concurrent tests must be >= 1"):
            invalid_spec = HardwareSpecification(concurrent_tests=0)
            invalid_spec.validate()

    def test_network_conditions_validation(self) -> None:
        """Test network conditions validation logic."""
        # Valid conditions
        valid_conditions = NetworkConditions(
            latency_ms=50,
            bandwidth_mbps=100,
            packet_loss_percent=0.5,
        )
        valid_conditions.validate()  # Should not raise

        # Invalid latency
        with pytest.raises(ValueError, match="Latency must be >= 0"):
            invalid_conditions = NetworkConditions(latency_ms=-1)
            invalid_conditions.validate()

        # Invalid bandwidth
        with pytest.raises(ValueError, match="Bandwidth must be > 0"):
            invalid_conditions = NetworkConditions(bandwidth_mbps=0)
            invalid_conditions.validate()

        # Invalid packet loss
        with pytest.raises(ValueError, match="Packet loss must be 0-100%"):
            invalid_conditions = NetworkConditions(packet_loss_percent=150.0)
            invalid_conditions.validate()

    def test_environment_isolation_validation(self) -> None:
        """Test environment isolation validation logic."""
        # Valid isolation
        valid_isolation = EnvironmentIsolation(
            port_range=(8600, 8699),
        )
        valid_isolation.validate()  # Should not raise

        # Invalid port range (start >= end)
        with pytest.raises(ValueError, match="Invalid port range"):
            invalid_isolation = EnvironmentIsolation(port_range=(8699, 8600))
            invalid_isolation.validate()

        # Invalid start port
        with pytest.raises(ValueError, match="Invalid start port"):
            invalid_isolation = EnvironmentIsolation(port_range=(500, 8699))
            invalid_isolation.validate()

        # Invalid end port
        with pytest.raises(ValueError, match="Invalid end port"):
            invalid_isolation = EnvironmentIsolation(port_range=(8600, 70000))
            invalid_isolation.validate()

    def test_test_environment_config_initialization(self) -> None:
        """Test TestEnvironmentConfig initialization and validation."""
        # Test with default values
        config = TestEnvironmentConfig()

        assert config.hardware is not None
        assert config.network is not None
        assert config.isolation is not None
        assert config.environment_id.startswith("crackseg_e2e_")
        assert config.test_session_id.startswith("session_")

        # Test with custom values
        custom_hardware = HardwareSpecification(cpu_cores=4, memory_mb=8192)
        custom_network = NetworkConditions(latency_ms=100)
        custom_isolation = EnvironmentIsolation(port_range=(9000, 9099))

        custom_config = TestEnvironmentConfig(
            hardware=custom_hardware,
            network=custom_network,
            isolation=custom_isolation,
        )

        assert custom_config.hardware.cpu_cores == 4
        assert custom_config.network.latency_ms == 100
        assert custom_config.isolation.port_range == (9000, 9099)

    def test_system_capability_checker(self) -> None:
        """Test system capability checking functionality."""
        checker = SystemCapabilityChecker()

        # Test hardware compatibility check
        spec = HardwareSpecification(
            cpu_cores=1,  # Minimal requirement
            memory_mb=1024,  # Minimal requirement
            disk_space_mb=1024,  # Minimal requirement
        )

        results = checker.check_hardware_compatibility(spec)

        assert "compatible" in results
        assert "warnings" in results
        assert "errors" in results
        assert "system_info" in results

        # Should have system info
        assert "cpu_cores" in results["system_info"]
        assert "memory_mb" in results["system_info"]
        assert "disk_free_mb" in results["system_info"]
        assert "platform" in results["system_info"]

        # Test network conditions check
        conditions = NetworkConditions(connection_timeout_sec=5)
        network_results = checker.check_network_conditions(conditions)

        assert "reachable" in network_results
        assert "latency_ok" in network_results
        assert "bandwidth_ok" in network_results
        assert "warnings" in network_results

    def test_test_environment_manager_initialization(self) -> None:
        """Test TestEnvironmentManager initialization."""
        # Test with default configuration
        manager = TestEnvironmentManager()

        assert manager.config is not None
        assert manager.resource_manager is not None
        assert manager.browser_config_manager is not None
        assert manager.env_manager is not None
        assert manager.capability_checker is not None

        # Test with custom configuration
        custom_config = TestEnvironmentConfig(
            hardware=HardwareSpecification(cpu_cores=2),
        )
        custom_manager = TestEnvironmentManager(config=custom_config)

        assert custom_manager.config.hardware.cpu_cores == 2

    def test_environment_validation(self) -> None:
        """Test environment validation functionality."""
        manager = TestEnvironmentManager()

        validation_results = manager.validate_environment()

        assert "valid" in validation_results
        assert "hardware" in validation_results
        assert "network" in validation_results
        assert "isolation" in validation_results
        assert "warnings" in validation_results
        assert "errors" in validation_results

        # Hardware validation should be present
        hardware_results = validation_results["hardware"]
        assert "compatible" in hardware_results
        assert "system_info" in hardware_results

        # Network validation should be present
        network_results = validation_results["network"]
        assert "reachable" in network_results

    def test_environment_status_tracking(self) -> None:
        """Test environment status tracking functionality."""
        manager = TestEnvironmentManager()

        # Initial status
        status = manager.get_environment_status()
        assert status["setup_complete"] is False
        assert status["active_allocations"] == 0
        assert status["temp_directories"] == 0

        # Environment ID should be set
        assert status["environment_id"].startswith("crackseg_e2e_")

    def test_environment_reset(self) -> None:
        """Test environment reset functionality."""
        manager = TestEnvironmentManager()

        original_env_id = manager.config.environment_id
        original_session_id = manager.config.test_session_id

        # Wait a moment to ensure different timestamps
        time.sleep(1)

        manager.reset_environment()

        # IDs should be different after reset
        assert manager.config.environment_id != original_env_id
        assert manager.config.test_session_id != original_session_id

        # Status should show clean state
        status = manager.get_environment_status()
        assert status["setup_complete"] is False
        assert status["active_allocations"] == 0


@pytest.mark.isolated_environment
class TestEnvironmentSetupIntegration:
    """Integration tests for test environment setup with fixtures."""

    def test_isolated_test_environment_fixture(
        self, isolated_test_environment: dict[str, Any]
    ) -> None:
        """Test the isolated test environment fixture functionality."""
        # Verify environment info structure
        assert "environment_id" in isolated_test_environment
        assert "test_session_id" in isolated_test_environment
        assert "resource_allocation" in isolated_test_environment
        assert "browser_matrix" in isolated_test_environment
        assert "validation_results" in isolated_test_environment
        assert "directories" in isolated_test_environment
        assert "network_conditions" in isolated_test_environment
        assert "hardware_specs" in isolated_test_environment

        # Verify environment setup is valid
        validation = isolated_test_environment["validation_results"]
        assert validation["valid"] is True

        # Verify directories exist
        directories = isolated_test_environment["directories"]
        artifacts_dir = Path(directories["artifacts"])
        temp_dir = Path(directories["temp"])

        assert artifacts_dir.exists()
        assert temp_dir.exists()

        logger.info(
            f"Environment ID: {isolated_test_environment['environment_id']}"
        )
        logger.info(f"Validation status: {validation['valid']}")

    @pytest.mark.performance_environment
    def test_performance_test_environment_fixture(
        self, performance_test_environment: dict[str, Any]
    ) -> None:
        """Test the performance test environment fixture functionality."""
        # Verify environment info structure
        assert "environment_id" in performance_test_environment
        assert "hardware_specs" in performance_test_environment

        # Verify performance-optimized settings
        hardware = performance_test_environment["hardware_specs"]
        assert hardware.cpu_cores >= 4
        assert hardware.memory_mb >= 8192
        assert hardware.browser_instances >= 4
        assert hardware.concurrent_tests >= 6

        logger.info(
            f"Performance environment configured with {hardware.cpu_cores} CPU cores"
        )
        logger.info(f"Memory allocation: {hardware.memory_mb}MB")

    def test_environment_validation_results_fixture(
        self, environment_validation_results: dict[str, Any]
    ) -> None:
        """Test the environment validation results fixture."""
        assert "valid" in environment_validation_results
        assert "hardware" in environment_validation_results
        assert "network" in environment_validation_results
        assert "isolation" in environment_validation_results

        # Log validation results for debugging
        logger.info(
            f"Environment validation: {environment_validation_results['valid']}"
        )
        if environment_validation_results["warnings"]:
            logger.warning(
                f"Validation warnings: {environment_validation_results['warnings']}"
            )

    def test_hardware_spec_fixtures(
        self,
        hardware_spec_default: HardwareSpecification,
        hardware_spec_high_performance: HardwareSpecification,
    ) -> None:
        """Test hardware specification fixtures."""
        # Verify default specification
        assert hardware_spec_default.cpu_cores == 2
        assert hardware_spec_default.memory_mb == 4096
        assert hardware_spec_default.gpu_memory_mb == 8192  # RTX 3070 Ti

        # Verify high-performance specification
        assert hardware_spec_high_performance.cpu_cores >= 4
        assert hardware_spec_high_performance.memory_mb >= 8192
        assert hardware_spec_high_performance.max_model_batch_size == 16

        logger.info(
            f"Default hardware: {hardware_spec_default.cpu_cores} cores, {hardware_spec_default.memory_mb}MB"
        )
        logger.info(
            f"Performance hardware: {hardware_spec_high_performance.cpu_cores} cores, {hardware_spec_high_performance.memory_mb}MB"
        )

    def test_network_conditions_fixtures(
        self,
        network_conditions_default: NetworkConditions,
        network_conditions_slow: NetworkConditions,
    ) -> None:
        """Test network conditions fixtures."""
        # Verify default conditions
        assert network_conditions_default.latency_ms == 50
        assert network_conditions_default.bandwidth_mbps == 100
        assert network_conditions_default.packet_loss_percent == 0.0

        # Verify slow conditions
        assert network_conditions_slow.latency_ms == 200
        assert network_conditions_slow.bandwidth_mbps == 10
        assert network_conditions_slow.packet_loss_percent == 1.0

        logger.info(
            f"Default network: {network_conditions_default.latency_ms}ms latency"
        )
        logger.info(
            f"Slow network: {network_conditions_slow.latency_ms}ms latency"
        )


@pytest.mark.hardware_validation
class TestHardwareCompatibility:
    """Tests for hardware compatibility validation."""

    def test_rtx_3070_ti_compatibility(self) -> None:
        """Test compatibility with RTX 3070 Ti constraints."""
        # RTX 3070 Ti specification
        rtx_3070_ti_spec = HardwareSpecification(
            cpu_cores=4,
            memory_mb=16384,  # 16GB system RAM (typical)
            gpu_memory_mb=8192,  # 8GB VRAM
            max_model_batch_size=16,  # Maximum recommended
            browser_instances=3,
            concurrent_tests=4,
        )

        checker = SystemCapabilityChecker()
        results = checker.check_hardware_compatibility(rtx_3070_ti_spec)

        # Log results for analysis
        logger.info(f"RTX 3070 Ti compatibility: {results['compatible']}")
        logger.info(f"System info: {results['system_info']}")

        if results["warnings"]:
            logger.warning(f"Compatibility warnings: {results['warnings']}")

        if results["errors"]:
            logger.error(f"Compatibility errors: {results['errors']}")

        # The test should complete without crashes, regardless of compatibility
        assert "compatible" in results
        assert "system_info" in results


@pytest.mark.network_validation
class TestNetworkConditions:
    """Tests for network conditions validation."""

    def test_baseline_network_conditions(self) -> None:
        """Test baseline network conditions setup."""
        baseline_conditions = NetworkConditions(
            latency_ms=50,
            bandwidth_mbps=100,
            packet_loss_percent=0.0,
            connection_timeout_sec=30,
        )

        checker = SystemCapabilityChecker()
        results = checker.check_network_conditions(baseline_conditions)

        logger.info(f"Network reachability: {results['reachable']}")
        logger.info(f"Network validation: {results}")

        # Basic validation
        assert "reachable" in results
        assert "warnings" in results

    def test_degraded_network_conditions(self) -> None:
        """Test degraded network conditions handling."""
        degraded_conditions = NetworkConditions(
            latency_ms=500,  # High latency
            bandwidth_mbps=1,  # Low bandwidth
            packet_loss_percent=5.0,  # Packet loss
            connection_timeout_sec=60,  # Extended timeout
        )

        checker = SystemCapabilityChecker()
        results = checker.check_network_conditions(degraded_conditions)

        logger.info(f"Degraded network validation: {results}")

        # Should handle degraded conditions gracefully
        assert "reachable" in results
