"""Integration tests for E2E Resource Cleanup Automation System."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.cleanup import (
    CleanupConfig,
    CleanupManager,
    CleanupValidator,
    ValidationStatus,
    validate_and_rollback,
)


class TestCleanupSystemIntegration:
    """Integration tests for the complete cleanup system."""

    @pytest.mark.asyncio
    async def test_full_cleanup_workflow(self):
        """Test complete workflow: baseline → cleanup → validation."""
        # Configure cleanup with realistic settings
        config = CleanupConfig(
            enable_resource_monitoring=False,  # Disable for testing
            cleanup_timeout=10.0,
            max_memory_leak_mb=100.0,
            force_cleanup=True,
        )

        # Initialize manager and establish baseline
        manager = CleanupManager(config)
        baseline = await manager.establish_baseline()

        assert baseline is not None
        assert baseline.timestamp > 0

        # Mock cleanup registry for test
        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = [
                "temp_files",
                "processes",
            ]
            mock_registry.execute_cleanup = AsyncMock(
                return_value=3  # Mock cleaned resources
            )

            # Execute cleanup
            test_id = "integration_test_001"
            result = await manager.execute_cleanup(
                test_id, ["temp_files", "processes"]
            )

            # Verify cleanup results
            assert result.status.value in ["success", "partial"]
            assert result.duration_seconds >= 0
            assert "temp_files" in result.resources_cleaned
            assert "processes" in result.resources_cleaned

    @pytest.mark.asyncio
    async def test_cleanup_with_validation(self):
        """Test cleanup with post-cleanup validation."""
        # Create test snapshots
        import time

        from crackseg.utils.monitoring import ResourceSnapshot

        baseline = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=10.0,
            memory_used_mb=1000.0,
            memory_available_mb=7000.0,
            memory_percent=12.5,
            gpu_memory_used_mb=500.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.1,
            gpu_utilization_percent=2.0,
            gpu_temperature_celsius=30.0,
            process_count=40,
            thread_count=150,
            file_handles=80,
            network_connections=10,
            open_ports=[],
            disk_read_mb=50.0,
            disk_write_mb=25.0,
            temp_files_count=0,
            temp_files_size_mb=0.0,
        )

        # Post-cleanup snapshot with acceptable resource usage
        post_cleanup = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=10.0,
            memory_used_mb=1020.0,  # Small increase within tolerance
            memory_available_mb=6980.0,
            memory_percent=12.8,
            gpu_memory_used_mb=500.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.1,
            gpu_utilization_percent=2.0,
            gpu_temperature_celsius=30.0,
            process_count=40,
            thread_count=150,
            file_handles=80,
            network_connections=10,
            open_ports=[],
            disk_read_mb=50.0,
            disk_write_mb=25.0,
            temp_files_count=0,
            temp_files_size_mb=0.0,
        )

        # Validate cleanup results
        validator = CleanupValidator()
        validation_result = await validator.validate_cleanup(
            baseline, post_cleanup, "test_validation_001"
        )

        assert validation_result.status == ValidationStatus.PASSED
        assert validation_result.leak_detected is False
        assert validation_result.validation_duration >= 0

    @pytest.mark.asyncio
    async def test_cleanup_with_rollback(self):
        """Test cleanup with rollback when validation fails."""
        # Create test snapshots with resource leak
        import time

        from crackseg.utils.monitoring import ResourceSnapshot

        baseline = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=10.0,
            memory_used_mb=1000.0,
            memory_available_mb=7000.0,
            memory_percent=12.5,
            gpu_memory_used_mb=500.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.1,
            gpu_utilization_percent=2.0,
            gpu_temperature_celsius=30.0,
            process_count=40,
            thread_count=150,
            file_handles=80,
            network_connections=10,
            open_ports=[],
            disk_read_mb=50.0,
            disk_write_mb=25.0,
            temp_files_count=0,
            temp_files_size_mb=0.0,
        )

        # Post-cleanup snapshot with significant resource leak
        post_cleanup = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=10.0,
            memory_used_mb=1200.0,  # 200MB increase - exceeds tolerance
            memory_available_mb=6800.0,
            memory_percent=15.0,
            gpu_memory_used_mb=500.0,
            gpu_memory_total_mb=8192.0,
            gpu_memory_percent=6.1,
            gpu_utilization_percent=2.0,
            gpu_temperature_celsius=30.0,
            process_count=50,  # 10 more processes - exceeds tolerance
            thread_count=180,
            file_handles=80,
            network_connections=10,
            open_ports=[],
            disk_read_mb=50.0,
            disk_write_mb=25.0,
            temp_files_count=0,
            temp_files_size_mb=0.0,
        )

        # Mock registry for rollback testing
        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.execute_cleanup = AsyncMock(
                return_value=2  # Mock rollback cleanup
            )

            # Test validation with automatic rollback
            validation_result = await validate_and_rollback(
                baseline,
                post_cleanup,
                "test_rollback_001",
                ["temp_files", "processes"],
            )

            # Verify rollback was triggered and executed
            assert validation_result.rollback_performed is True
            assert validation_result.status in [
                ValidationStatus.ROLLBACK_REQUIRED,
                ValidationStatus.ROLLBACK_COMPLETED,
            ]

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_procedure(self):
        """Test actual temp file cleanup with real files."""
        # Create temporary directory and files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test files
            test_files = [
                test_dir / "temp_test_001.tmp",
                test_dir / "crackseg_test_output.txt",
                test_dir / "test_temp_file.temp",
                test_dir / "normal_file.txt",  # Should not be cleaned
            ]

            for file_path in test_files:
                file_path.write_text("test content")

            # Verify all files exist
            assert all(f.exists() for f in test_files)

            # Import and test temp file cleanup
            from tests.e2e.cleanup.resource_cleanup import TempFileCleanup

            cleanup = TempFileCleanup(
                {
                    "temp_patterns": ["temp_*", "*.tmp", "crackseg_test_*"],
                    "temp_directories": [str(test_dir)],
                }
            )

            # Execute cleanup
            cleaned_count = await cleanup.cleanup("test_temp_cleanup")

            # Verify cleanup results
            assert cleaned_count >= 3  # At least 3 temp files cleaned

            # Verify temp files are gone but normal file remains
            assert not (test_dir / "temp_test_001.tmp").exists()
            assert not (test_dir / "crackseg_test_output.txt").exists()
            assert not (test_dir / "test_temp_file.temp").exists()
            assert (test_dir / "normal_file.txt").exists()  # Preserved

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self):
        """Test cleanup system performance with multiple operations."""
        config = CleanupConfig(
            enable_resource_monitoring=False,
            cleanup_timeout=5.0,
            max_retries=2,
        )

        manager = CleanupManager(config)

        # Mock cleanup operations
        with patch(
            "tests.e2e.cleanup.resource_cleanup.ResourceCleanupRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_available_procedures.return_value = [
                "temp_files"
            ]
            mock_registry.execute_cleanup = AsyncMock(return_value=1)

            # Execute multiple cleanup operations sequentially
            # (concurrent operations are prevented by design)
            start_time = asyncio.get_event_loop().time()

            for i in range(5):
                result = await manager.execute_cleanup(f"perf_test_{i:03d}")
                assert result.status.value in ["success", "partial"]

            total_time = asyncio.get_event_loop().time() - start_time

            # Performance check: should complete reasonably quickly
            assert total_time < 30.0  # 5 operations in under 30 seconds

            # Each operation should be efficient
            avg_time_per_operation = total_time / 5
            assert avg_time_per_operation < 6.0  # Average under 6 seconds
