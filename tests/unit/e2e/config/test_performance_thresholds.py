"""Unit tests for performance thresholds system.

Tests for type-safe threshold models, configuration loading,
and validation logic for E2E testing pipeline performance.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.core.exceptions import ValidationError
from tests.e2e.config.performance_thresholds import (
    ContainerManagementThresholds,
    FileOperationThresholds,
    ModelProcessingThresholds,
    PerformanceThresholds,
    SystemResourceThresholds,
    WebInterfaceThresholds,
)
from tests.e2e.config.threshold_validator import (
    ThresholdValidator,
    ThresholdViolation,
    ViolationSeverity,
)


class TestWebInterfaceThresholds:
    """Test WebInterfaceThresholds dataclass validation."""

    def test_valid_configuration(self) -> None:
        """Test creation with valid threshold values."""
        thresholds = WebInterfaceThresholds(
            page_load_warning_ms=1500,
            page_load_critical_ms=2000,
            page_load_timeout_ms=5000,
            config_validation_warning_ms=300,
            config_validation_critical_ms=500,
            config_validation_timeout_ms=2000,
            button_response_ms=200,
            form_submission_ms=1000,
            file_upload_timeout_ms=10000,
        )

        assert thresholds.page_load_warning_ms == 1500
        assert thresholds.page_load_critical_ms == 2000
        assert thresholds.config_validation_warning_ms == 300

    def test_invalid_page_load_thresholds(self) -> None:
        """Test validation fails when warning >= critical."""
        with pytest.raises(
            ValidationError,
            match="warning threshold must be less than critical",
        ):
            WebInterfaceThresholds(
                page_load_warning_ms=2000,  # Same as critical
                page_load_critical_ms=2000,
                page_load_timeout_ms=5000,
                config_validation_warning_ms=300,
                config_validation_critical_ms=500,
                config_validation_timeout_ms=2000,
                button_response_ms=200,
                form_submission_ms=1000,
                file_upload_timeout_ms=10000,
            )

    def test_invalid_timeout_threshold(self) -> None:
        """Test validation fails when critical >= timeout."""
        with pytest.raises(
            ValidationError,
            match="critical threshold must be less than timeout",
        ):
            WebInterfaceThresholds(
                page_load_warning_ms=1500,
                page_load_critical_ms=5000,  # Same as timeout
                page_load_timeout_ms=5000,
                config_validation_warning_ms=300,
                config_validation_critical_ms=500,
                config_validation_timeout_ms=2000,
                button_response_ms=200,
                form_submission_ms=1000,
                file_upload_timeout_ms=10000,
            )

    def test_from_config_creation(self) -> None:
        """Test creation from mock configuration object."""
        mock_config = Mock()
        mock_config.web_interface.page_load_time.warning_threshold_ms = 1500
        mock_config.web_interface.page_load_time.critical_threshold_ms = 2000
        mock_config.web_interface.page_load_time.timeout_ms = 5000
        mock_config.web_interface.config_validation.warning_threshold_ms = 300
        mock_config.web_interface.config_validation.critical_threshold_ms = 500
        mock_config.web_interface.config_validation.timeout_ms = 2000
        mock_config.web_interface.user_interaction.button_response_ms = 200
        mock_config.web_interface.user_interaction.form_submission_ms = 1000
        mock_config.web_interface.user_interaction.file_upload_timeout_ms = (
            10000
        )

        thresholds = WebInterfaceThresholds.from_config(mock_config)

        assert thresholds.page_load_warning_ms == 1500
        assert thresholds.button_response_ms == 200


class TestModelProcessingThresholds:
    """Test ModelProcessingThresholds dataclass validation."""

    def test_valid_configuration(self) -> None:
        """Test creation with valid threshold values."""
        thresholds = ModelProcessingThresholds(
            inference_warning_ms=800,
            inference_critical_ms=1200,
            batch_timeout_ms=5000,
            memory_warning_mb=6000,
            memory_critical_mb=7500,
            oom_threshold_mb=7800,
            min_precision=0.85,
            min_recall=0.80,
            min_iou=0.75,
        )

        assert thresholds.inference_warning_ms == 800
        assert thresholds.memory_warning_mb == 6000
        assert thresholds.min_precision == 0.85

    def test_invalid_memory_thresholds(self) -> None:
        """Test validation fails with invalid memory thresholds."""
        with pytest.raises(
            ValidationError,
            match="Memory warning threshold must be less than critical",
        ):
            ModelProcessingThresholds(
                inference_warning_ms=800,
                inference_critical_ms=1200,
                batch_timeout_ms=5000,
                memory_warning_mb=7500,  # Same as critical
                memory_critical_mb=7500,
                oom_threshold_mb=7800,
                min_precision=0.85,
                min_recall=0.80,
                min_iou=0.75,
            )

    def test_invalid_quality_metrics(self) -> None:
        """Test validation fails with quality metrics outside [0,1] range."""
        with pytest.raises(
            ValidationError,
            match="Minimum precision must be between 0.0 and 1.0",
        ):
            ModelProcessingThresholds(
                inference_warning_ms=800,
                inference_critical_ms=1200,
                batch_timeout_ms=5000,
                memory_warning_mb=6000,
                memory_critical_mb=7500,
                oom_threshold_mb=7800,
                min_precision=1.5,  # Invalid: > 1.0
                min_recall=0.80,
                min_iou=0.75,
            )

    def test_quality_metrics_boundary_values(self) -> None:
        """Test quality metrics accept boundary values 0.0 and 1.0."""
        thresholds = ModelProcessingThresholds(
            inference_warning_ms=800,
            inference_critical_ms=1200,
            batch_timeout_ms=5000,
            memory_warning_mb=6000,
            memory_critical_mb=7500,
            oom_threshold_mb=7800,
            min_precision=1.0,  # Valid boundary
            min_recall=0.0,  # Valid boundary
            min_iou=0.75,
        )

        assert thresholds.min_precision == 1.0
        assert thresholds.min_recall == 0.0


class TestSystemResourceThresholds:
    """Test SystemResourceThresholds dataclass validation."""

    def test_valid_configuration(self) -> None:
        """Test creation with valid threshold values."""
        thresholds = SystemResourceThresholds(
            cpu_warning_percent=75,
            cpu_critical_percent=90,
            cpu_sustained_duration_s=30,
            memory_warning_mb=8000,
            memory_critical_mb=12000,
            memory_leak_growth_mb=1000,
            temp_files_warning_mb=500,
            temp_files_critical_mb=1000,
            log_files_max_mb=100,
        )

        assert thresholds.cpu_warning_percent == 75
        assert thresholds.memory_warning_mb == 8000

    def test_invalid_cpu_percentage_range(self) -> None:
        """Test validation fails with CPU percentage outside [0,100] range."""
        with pytest.raises(
            ValidationError,
            match="CPU warning threshold must be less than critical threshold",
        ):
            SystemResourceThresholds(
                cpu_warning_percent=150,  # Invalid: > 100
                cpu_critical_percent=90,
                cpu_sustained_duration_s=30,
                memory_warning_mb=8000,
                memory_critical_mb=12000,
                memory_leak_growth_mb=1000,
                temp_files_warning_mb=500,
                temp_files_critical_mb=1000,
                log_files_max_mb=100,
            )

    def test_cpu_percentage_boundary_values(self) -> None:
        """Test CPU percentages accept boundary values 0 and 100."""
        thresholds = SystemResourceThresholds(
            cpu_warning_percent=0,  # Valid boundary
            cpu_critical_percent=100,  # Valid boundary
            cpu_sustained_duration_s=30,
            memory_warning_mb=8000,
            memory_critical_mb=12000,
            memory_leak_growth_mb=1000,
            temp_files_warning_mb=500,
            temp_files_critical_mb=1000,
            log_files_max_mb=100,
        )

        assert thresholds.cpu_warning_percent == 0
        assert thresholds.cpu_critical_percent == 100


class TestContainerManagementThresholds:
    """Test ContainerManagementThresholds dataclass validation."""

    def test_valid_configuration(self) -> None:
        """Test creation with valid threshold values."""
        thresholds = ContainerManagementThresholds(
            startup_warning_s=15,
            startup_critical_s=30,
            startup_timeout_s=60,
            shutdown_warning_s=5,
            shutdown_critical_s=10,
            force_kill_timeout_s=30,
            max_orphaned_containers=2,
            max_dangling_images=5,
            max_unused_volumes=3,
        )

        assert thresholds.startup_warning_s == 15
        assert thresholds.max_orphaned_containers == 2

    def test_invalid_startup_thresholds(self) -> None:
        """Test validation fails with invalid startup timing."""
        with pytest.raises(
            ValidationError,
            match="Container startup warning must be less than critical",
        ):
            ContainerManagementThresholds(
                startup_warning_s=30,  # Same as critical
                startup_critical_s=30,
                startup_timeout_s=60,
                shutdown_warning_s=5,
                shutdown_critical_s=10,
                force_kill_timeout_s=30,
                max_orphaned_containers=2,
                max_dangling_images=5,
                max_unused_volumes=3,
            )


class TestFileOperationThresholds:
    """Test FileOperationThresholds dataclass validation."""

    def test_valid_configuration(self) -> None:
        """Test creation with valid threshold values."""
        thresholds = FileOperationThresholds(
            read_warning_ms=100,
            read_critical_ms=500,
            write_warning_ms=200,
            write_critical_ms=1000,
            screenshot_save_ms=1000,
            report_generation_ms=3000,
            cleanup_completion_ms=2000,
            max_open_files=100,
            lock_timeout_ms=5000,
        )

        assert thresholds.read_warning_ms == 100
        assert thresholds.max_open_files == 100

    def test_invalid_max_open_files(self) -> None:
        """Test validation fails with non-positive max_open_files."""
        with pytest.raises(
            ValidationError, match="Maximum open files must be positive"
        ):
            FileOperationThresholds(
                read_warning_ms=100,
                read_critical_ms=500,
                write_warning_ms=200,
                write_critical_ms=1000,
                screenshot_save_ms=1000,
                report_generation_ms=3000,
                cleanup_completion_ms=2000,
                max_open_files=0,  # Invalid: must be positive
                lock_timeout_ms=5000,
            )


class TestPerformanceThresholds:
    """Test master PerformanceThresholds container."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create mock configuration for testing."""
        config = Mock()

        # Web interface config
        config.web_interface.page_load_time.warning_threshold_ms = 1500
        config.web_interface.page_load_time.critical_threshold_ms = 2000
        config.web_interface.page_load_time.timeout_ms = 5000
        config.web_interface.config_validation.warning_threshold_ms = 300
        config.web_interface.config_validation.critical_threshold_ms = 500
        config.web_interface.config_validation.timeout_ms = 2000
        config.web_interface.user_interaction.button_response_ms = 200
        config.web_interface.user_interaction.form_submission_ms = 1000
        config.web_interface.user_interaction.file_upload_timeout_ms = 10000

        # Model processing config
        config.model_processing.inference_time.warning_threshold_ms = 800
        config.model_processing.inference_time.critical_threshold_ms = 1200
        config.model_processing.inference_time.batch_timeout_ms = 5000
        config.model_processing.memory_usage.warning_threshold_mb = 6000
        config.model_processing.memory_usage.critical_threshold_mb = 7500
        config.model_processing.memory_usage.oom_threshold_mb = 7800
        config.model_processing.crack_detection.min_precision = 0.85
        config.model_processing.crack_detection.min_recall = 0.80
        config.model_processing.crack_detection.min_iou = 0.75

        # System resources config
        config.system_resources.cpu_usage.warning_threshold_percent = 75
        config.system_resources.cpu_usage.critical_threshold_percent = 90
        config.system_resources.cpu_usage.sustained_duration_s = 30
        config.system_resources.memory_usage.warning_threshold_mb = 8000
        config.system_resources.memory_usage.critical_threshold_mb = 12000
        config.system_resources.memory_usage.leak_detection_growth_mb = 1000
        config.system_resources.disk_usage.temp_files_warning_mb = 500
        config.system_resources.disk_usage.temp_files_critical_mb = 1000
        config.system_resources.disk_usage.log_files_max_mb = 100

        # Container management config
        config.container_management.startup_time.warning_threshold_s = 15
        config.container_management.startup_time.critical_threshold_s = 30
        config.container_management.startup_time.timeout_s = 60
        config.container_management.shutdown_time.warning_threshold_s = 5
        config.container_management.shutdown_time.critical_threshold_s = 10
        config.container_management.shutdown_time.force_kill_timeout_s = 30
        (
            config.container_management.resource_cleanup.orphaned_containers_max
        ) = 2
        config.container_management.resource_cleanup.dangling_images_max = 5
        config.container_management.resource_cleanup.unused_volumes_max = 3

        # File operations config
        config.file_operations.test_data_access.read_time_warning_ms = 100
        config.file_operations.test_data_access.read_time_critical_ms = 500
        config.file_operations.test_data_access.write_time_warning_ms = 200
        config.file_operations.test_data_access.write_time_critical_ms = 1000
        config.file_operations.artifact_generation.screenshot_save_ms = 1000
        config.file_operations.artifact_generation.report_generation_ms = 3000
        config.file_operations.artifact_generation.cleanup_completion_ms = 2000
        config.file_operations.concurrent_access.max_open_files = 100
        config.file_operations.concurrent_access.lock_timeout_ms = 5000

        return config

    def test_from_config_creation(self, mock_config: Mock) -> None:
        """Test creating PerformanceThresholds from configuration."""
        thresholds = PerformanceThresholds.from_config(mock_config)

        assert isinstance(thresholds.web_interface, WebInterfaceThresholds)
        assert isinstance(
            thresholds.model_processing, ModelProcessingThresholds
        )
        assert isinstance(
            thresholds.system_resources, SystemResourceThresholds
        )
        assert isinstance(
            thresholds.container_management, ContainerManagementThresholds
        )
        assert isinstance(thresholds.file_operations, FileOperationThresholds)

        # Verify values propagated correctly
        assert thresholds.web_interface.page_load_warning_ms == 1500
        assert thresholds.model_processing.memory_warning_mb == 6000

    def test_validation_success(self, mock_config: Mock) -> None:
        """Test successful validation of complete threshold configuration."""
        thresholds = PerformanceThresholds.from_config(mock_config)

        # Should not raise exception
        thresholds.validate()

    def test_get_summary(self, mock_config: Mock) -> None:
        """Test summary generation for key thresholds."""
        thresholds = PerformanceThresholds.from_config(mock_config)
        summary = thresholds.get_summary()

        expected_keys = {
            "page_load_sla_ms",
            "config_validation_sla_ms",
            "inference_sla_ms",
            "vram_limit_mb",
            "cpu_limit_percent",
            "container_startup_sla_s",
        }

        assert set(summary.keys()) == expected_keys
        assert summary["page_load_sla_ms"] == "2000"
        assert summary["vram_limit_mb"] == "7500"

    def test_invalid_config_handling(self) -> None:
        """Test handling of invalid configuration data."""
        invalid_config = Mock()
        invalid_config.web_interface.page_load_time.warning_threshold_ms = 2000
        invalid_config.web_interface.page_load_time.critical_threshold_ms = (
            1500  # Invalid: warning > critical
        )

        with pytest.raises(
            ValidationError, match="Failed to load performance thresholds"
        ):
            PerformanceThresholds.from_config(invalid_config)


class TestThresholdValidator:
    """Test ThresholdValidator functionality."""

    @pytest.fixture
    def sample_thresholds(self) -> PerformanceThresholds:
        """Create sample performance thresholds for testing."""
        return PerformanceThresholds(
            web_interface=WebInterfaceThresholds(
                page_load_warning_ms=1500,
                page_load_critical_ms=2000,
                page_load_timeout_ms=5000,
                config_validation_warning_ms=300,
                config_validation_critical_ms=500,
                config_validation_timeout_ms=2000,
                button_response_ms=200,
                form_submission_ms=1000,
                file_upload_timeout_ms=10000,
            ),
            model_processing=ModelProcessingThresholds(
                inference_warning_ms=800,
                inference_critical_ms=1200,
                batch_timeout_ms=5000,
                memory_warning_mb=6000,
                memory_critical_mb=7500,
                oom_threshold_mb=7800,
                min_precision=0.85,
                min_recall=0.80,
                min_iou=0.75,
            ),
            system_resources=SystemResourceThresholds(
                cpu_warning_percent=75,
                cpu_critical_percent=90,
                cpu_sustained_duration_s=30,
                memory_warning_mb=8000,
                memory_critical_mb=12000,
                memory_leak_growth_mb=1000,
                temp_files_warning_mb=500,
                temp_files_critical_mb=1000,
                log_files_max_mb=100,
            ),
            container_management=ContainerManagementThresholds(
                startup_warning_s=15,
                startup_critical_s=30,
                startup_timeout_s=60,
                shutdown_warning_s=5,
                shutdown_critical_s=10,
                force_kill_timeout_s=30,
                max_orphaned_containers=2,
                max_dangling_images=5,
                max_unused_volumes=3,
            ),
            file_operations=FileOperationThresholds(
                read_warning_ms=100,
                read_critical_ms=500,
                write_warning_ms=200,
                write_critical_ms=1000,
                screenshot_save_ms=1000,
                report_generation_ms=3000,
                cleanup_completion_ms=2000,
                max_open_files=100,
                lock_timeout_ms=5000,
            ),
        )

    @pytest.fixture
    def mock_monitoring_manager(self) -> Mock:
        """Create mock MonitoringManager for testing."""
        manager = Mock()
        manager.get_history.return_value = {
            "test/page_load_time_ms_values": [
                1000,
                1800,
                2200,
            ],  # Critical violation
            "test/config_validation_time_ms_values": [
                250,
                350,
            ],  # Warning violation
            "test/inference_time_ms_values": [700],  # No violation
            "test/gpu_memory_used_mb_values": [7600],  # Critical violation
            "test/cpu_usage_percent_values": [80],  # Warning violation
        }
        return manager

    def test_validator_initialization(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test ThresholdValidator initialization."""
        validator = ThresholdValidator(sample_thresholds)

        assert validator.thresholds == sample_thresholds
        assert validator.violations_history == []
        assert validator._last_validation_time == 0.0

    def test_validate_metrics_no_violations(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test validation with metrics below thresholds."""
        validator = ThresholdValidator(sample_thresholds)

        mock_manager = Mock()
        mock_manager.get_history.return_value = {
            "test/page_load_time_ms_values": [1000],  # Below warning (1500)
            "test/inference_time_ms_values": [600],  # Below warning (800)
            "test/cpu_usage_percent_values": [50],  # Below warning (75)
        }

        violations = validator.validate_metrics(mock_manager)

        assert violations == []
        assert len(validator.violations_history) == 0

    def test_validate_metrics_warning_violations(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test validation with warning-level violations."""
        validator = ThresholdValidator(sample_thresholds)

        mock_manager = Mock()
        mock_manager.get_history.return_value = {
            "test/page_load_time_ms_values": [
                1600
            ],  # Warning (1500 < 1600 < 2000)
            "test/cpu_usage_percent_values": [80],  # Warning (75 < 80 < 90)
        }

        violations = validator.validate_metrics(mock_manager)

        assert len(violations) == 2
        assert all(v.severity == ViolationSeverity.WARNING for v in violations)
        assert any(v.metric_name == "page_load_time_ms" for v in violations)
        assert any(v.metric_name == "cpu_usage_percent" for v in violations)

    def test_validate_metrics_critical_violations(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test validation with critical-level violations."""
        validator = ThresholdValidator(sample_thresholds)

        mock_manager = Mock()
        mock_manager.get_history.return_value = {
            "test/page_load_time_ms_values": [2500],  # Critical (>= 2000)
            "test/gpu_memory_used_mb_values": [7600],  # Critical (>= 7500)
        }

        violations = validator.validate_metrics(mock_manager)

        assert len(violations) == 2
        assert all(
            v.severity == ViolationSeverity.CRITICAL for v in violations
        )
        assert any(v.metric_name == "page_load_time_ms" for v in violations)
        assert any(v.metric_name == "gpu_memory_used_mb" for v in violations)

    def test_violations_summary_healthy(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test violations summary with no violations."""
        validator = ThresholdValidator(sample_thresholds)

        summary = validator.get_violations_summary()

        assert summary["total_violations"] == 0
        assert summary["status"] == "healthy"

    def test_violations_summary_with_history(
        self,
        sample_thresholds: PerformanceThresholds,
        mock_monitoring_manager: Mock,
    ) -> None:
        """Test violations summary with violation history."""
        validator = ThresholdValidator(sample_thresholds)

        # Generate some violations
        validator.validate_metrics(mock_monitoring_manager)

        summary = validator.get_violations_summary()

        assert summary["total_violations"] > 0
        assert "severity_breakdown" in summary
        assert "recent_violations_count" in summary
        assert summary["status"] in ["degraded", "stable"]

    def test_clear_violations_history(
        self, sample_thresholds: PerformanceThresholds
    ) -> None:
        """Test clearing violations history."""
        validator = ThresholdValidator(sample_thresholds)

        # Add some violations
        validator.violations_history = [
            ThresholdViolation(
                metric_name="test_metric",
                actual_value=100,
                threshold_value=50,
                severity=ViolationSeverity.WARNING,
                timestamp=123456789.0,
                context="test",
                message="Test violation",
            )
        ]

        validator.clear_violations_history()

        assert validator.violations_history == []

    @patch("tests.e2e.config.threshold_validator.initialize_config_dir")
    @patch("tests.e2e.config.threshold_validator.compose")
    def test_from_config_file_success(
        self,
        mock_compose: Mock,
        mock_initialize: Mock,
        sample_thresholds: PerformanceThresholds,
    ) -> None:
        """Test creating validator from configuration file."""
        # Mock Hydra configuration loading
        mock_compose.return_value = Mock()

        with patch(
            "tests.e2e.config.threshold_validator.PerformanceThresholds.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = sample_thresholds

            with tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False
            ) as temp_file:
                config_path = temp_file.name

            try:
                validator = ThresholdValidator.from_config_file(config_path)
                assert validator.thresholds == sample_thresholds
            finally:
                Path(config_path).unlink()  # Clean up

    def test_from_config_file_not_found(self) -> None:
        """Test error handling when configuration file doesn't exist."""
        non_existent_path = "/non/existent/path/config.yaml"

        with pytest.raises(
            ValidationError, match="Configuration file not found"
        ):
            ThresholdValidator.from_config_file(non_existent_path)
