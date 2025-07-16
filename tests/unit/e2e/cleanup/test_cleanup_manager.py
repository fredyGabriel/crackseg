"""Unit tests for CleanupManager with ResourceMonitor integration."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crackseg.utils.monitoring import ResourceSnapshot
from tests.e2e.cleanup import (
    CleanupConfig,
    CleanupManager,
    CleanupStatus,
)


@pytest.fixture
def mock_resource_monitor():
    """Create mock ResourceMonitor instance."""
    monitor = MagicMock()

    # Create a proper ResourceSnapshot with all required fields
    mock_snapshot = ResourceSnapshot(
        timestamp=time.time(),
        cpu_percent=15.0,
        memory_used_mb=1024.0,
        memory_available_mb=6144.0,
        memory_percent=14.3,
        gpu_memory_used_mb=512.0,
        gpu_memory_total_mb=8192.0,
        gpu_memory_percent=6.25,
        gpu_utilization_percent=5.0,
        gpu_temperature_celsius=35.0,
        process_count=45,
        thread_count=156,
        file_handles=89,
        network_connections=12,
        open_ports=[8000, 8080],
        disk_read_mb=100.0,
        disk_write_mb=50.0,
        temp_files_count=5,
        temp_files_size_mb=25.0,
    )

    monitor.get_current_snapshot.return_value = mock_snapshot
    monitor.start_monitoring = AsyncMock()
    monitor.stop_monitoring = AsyncMock()

    return monitor


@pytest.fixture
def cleanup_config():
    """Create test cleanup configuration."""
    return CleanupConfig(
        enable_resource_monitoring=True,
        cleanup_timeout=30.0,
        max_memory_leak_mb=50.0,
        max_process_leak_count=3,
        max_file_handle_leak_count=10,
    )


@pytest.fixture
def cleanup_manager(
    cleanup_config: Any, mock_resource_monitor: Any
) -> CleanupManager:
    """Create CleanupManager instance with mocked dependencies."""
    manager = CleanupManager(cleanup_config)
    manager.resource_monitor = mock_resource_monitor
    return manager


class TestCleanupManager:
    """Test suite for CleanupManager."""

    @pytest.mark.asyncio
    async def test_establish_baseline_with_monitoring(
        self, cleanup_manager: Any, mock_resource_monitor: Any
    ) -> None:
        """Test baseline establishment with resource monitoring enabled."""
        baseline = await cleanup_manager.establish_baseline()

        assert baseline is not None
        assert baseline.timestamp > 0
        assert baseline.memory_used_mb == 1024.0
        assert baseline.process_count == 45

        mock_resource_monitor.get_current_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_establish_baseline_without_monitoring(
        self, cleanup_config: Any
    ) -> None:
        """Test baseline establishment with monitoring disabled."""
        cleanup_config.enable_resource_monitoring = False
        manager = CleanupManager(cleanup_config)

        baseline = await manager.establish_baseline()

        assert baseline is not None
        assert baseline.cpu_percent == 0.0
        assert baseline.memory_used_mb == 0.0
        assert baseline.gpu_memory_total_mb == 8192.0  # RTX 3070 Ti default

    @pytest.mark.asyncio
    async def test_execute_cleanup_success(self, cleanup_manager: Any) -> None:
        """Test successful cleanup execution."""
        test_id = "test_cleanup_001"

        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = [
                "temp_files",
                "processes",
                "network_connections",
            ]
            mock_registry.execute_cleanup = AsyncMock(return_value=5)

            result = await cleanup_manager.execute_cleanup(test_id)

            assert result.status == CleanupStatus.SUCCESS
            assert result.duration_seconds > 0
            assert len(result.resources_cleaned) == 5  # Default procedures
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_cleanup_with_custom_procedures(
        self, cleanup_manager: Any
    ) -> None:
        """Test cleanup with custom procedure list."""
        test_id = "test_cleanup_002"
        custom_procedures = ["temp_files", "gpu_cache"]

        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = (
                custom_procedures
            )
            mock_registry.execute_cleanup = AsyncMock(return_value=3)

            result = await cleanup_manager.execute_cleanup(
                test_id, custom_procedures
            )

            assert result.status == CleanupStatus.SUCCESS
            assert len(result.resources_cleaned) == 2

    @pytest.mark.asyncio
    async def test_execute_cleanup_timeout(self, cleanup_manager: Any) -> None:
        """Test cleanup timeout handling."""
        test_id = "test_cleanup_timeout"
        cleanup_manager.config.cleanup_timeout = 0.1  # Very short timeout

        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = [
                "temp_files"
            ]

            # Simulate slow cleanup
            async def slow_cleanup(*args: Any, **kwargs: Any) -> int:
                await asyncio.sleep(0.2)
                return 1

            mock_registry.execute_cleanup = slow_cleanup

            result = await cleanup_manager.execute_cleanup(test_id)

            assert result.status == CleanupStatus.FAILED
            assert "timeout" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_cleanup_concurrent_execution_prevention(
        self, cleanup_manager: Any
    ) -> None:
        """Test that concurrent cleanup executions are prevented."""
        test_id = "test_concurrent"

        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = [
                "temp_files"
            ]

            # Simulate slow cleanup
            async def slow_cleanup(*args: Any, **kwargs: Any) -> int:
                await asyncio.sleep(0.1)
                return 1

            mock_registry.execute_cleanup = slow_cleanup

            # Start first cleanup
            task1 = asyncio.create_task(
                cleanup_manager.execute_cleanup(test_id)
            )

            # Try to start second cleanup while first is running
            await asyncio.sleep(0.01)  # Let first cleanup start

            with pytest.raises(RuntimeError, match="already in progress"):
                await cleanup_manager.execute_cleanup(test_id + "_2")

            # Wait for first cleanup to complete
            result1 = await task1
            assert result1.status == CleanupStatus.SUCCESS

    def test_detect_resource_leaks_memory(self, cleanup_manager: Any) -> None:
        """Test memory leak detection."""
        # baseline = cleanup_manager.baseline_snapshot  # Not used in this test

        # Create post-cleanup snapshot with memory leak
        post_cleanup = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=15.0,
            memory_used_mb=1124.0,  # 100MB more than baseline
            memory_available_mb=6044.0,
            memory_percent=15.5,
            gpu_memory_used_mb=512.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.25,
            gpu_utilization_percent=5.0,
            gpu_temperature_celsius=35.0,
            process_count=45,
            thread_count=156,
            file_handles=89,
            network_connections=12,
            open_ports=[8000, 8080],
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            temp_files_count=5,
            temp_files_size_mb=25.0,
        )

        # Memory leak exceeds threshold (50MB)
        leak_detected = cleanup_manager._detect_resource_leaks(post_cleanup)
        assert leak_detected is True

    def test_detect_resource_leaks_processes(
        self, cleanup_manager: Any
    ) -> None:
        """Test process leak detection."""
        # baseline = cleanup_manager.baseline_snapshot  # Not used in this test

        # Create post-cleanup snapshot with process leak
        post_cleanup = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=15.0,
            memory_used_mb=1024.0,
            memory_available_mb=6144.0,
            memory_percent=14.3,
            gpu_memory_used_mb=512.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.25,
            gpu_utilization_percent=5.0,
            gpu_temperature_celsius=35.0,
            process_count=50,  # 5 more processes than baseline
            thread_count=156,
            file_handles=89,
            network_connections=12,
            open_ports=[8000, 8080],
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            temp_files_count=5,
            temp_files_size_mb=25.0,
        )

        # Process leak exceeds threshold (3 processes)
        leak_detected = cleanup_manager._detect_resource_leaks(post_cleanup)
        assert leak_detected is True

    def test_no_resource_leaks(self, cleanup_manager: Any) -> None:
        """Test when no resource leaks are detected."""
        # baseline = cleanup_manager.baseline_snapshot  # Not used in this test

        # Create post-cleanup snapshot similar to baseline
        post_cleanup = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=15.0,
            memory_used_mb=1030.0,  # Small increase within tolerance
            memory_available_mb=6138.0,
            memory_percent=14.4,
            gpu_memory_used_mb=512.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.25,
            gpu_utilization_percent=5.0,
            gpu_temperature_celsius=35.0,
            process_count=45,  # Same as baseline
            thread_count=156,
            file_handles=89,
            network_connections=12,
            open_ports=[8000, 8080],
            disk_read_mb=100.0,
            disk_write_mb=50.0,
            temp_files_count=5,
            temp_files_size_mb=25.0,
        )

        leak_detected = cleanup_manager._detect_resource_leaks(post_cleanup)
        assert leak_detected is False

    def test_determine_cleanup_status_combinations(
        self, cleanup_manager: Any
    ) -> None:
        """Test cleanup status determination logic."""
        # Success case
        status = cleanup_manager._determine_cleanup_status({}, [], False)
        assert status == CleanupStatus.SUCCESS

        # Failed with leak
        status = cleanup_manager._determine_cleanup_status({}, ["error"], True)
        assert status == CleanupStatus.FAILED

        # Partial success (errors but no leak)
        status = cleanup_manager._determine_cleanup_status(
            {}, ["error"], False
        )
        assert status == CleanupStatus.PARTIAL

        # Rollback required (no errors but leak detected)
        status = cleanup_manager._determine_cleanup_status({}, [], True)
        assert status == CleanupStatus.ROLLBACK_REQUIRED

        # Success with resources cleaned
        status = cleanup_manager._determine_cleanup_status(
            {"temp_files": 5}, [], False
        )
        assert status == CleanupStatus.SUCCESS
