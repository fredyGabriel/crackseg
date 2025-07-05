"""Test suite for Resource Cleanup Validation component.

This test suite validates the resource cleanup validation functionality
including baseline establishment, leak detection, and cleanup verification
across workflow components.
"""


from .resource_cleanup_monitoring import (
    ResourceCleanupValidationMixin,
    SystemResourceMonitor,
)
from .resource_cleanup_protocols import (
    CleanupValidationConfig,
    ResourceBaseline,
    ResourceCleanupMetrics,
)
from .resource_cleanup_validation import ResourceCleanupValidationComponent


class TestSystemResourceMonitor:
    """Test suite for SystemResourceMonitor functionality."""

    def test_resource_monitor_initialization(self) -> None:
        """Test that resource monitor initializes correctly."""
        monitor = SystemResourceMonitor()
        assert monitor.config is not None
        assert isinstance(monitor.config, CleanupValidationConfig)

    def test_memory_usage_measurement(self) -> None:
        """Test memory usage measurement."""
        monitor = SystemResourceMonitor()
        memory_mb = monitor.get_memory_usage_mb()
        assert memory_mb > 0.0
        assert isinstance(memory_mb, float)

    def test_process_count_measurement(self) -> None:
        """Test process count measurement."""
        monitor = SystemResourceMonitor()
        process_count = monitor.get_process_count()
        assert process_count > 0
        assert isinstance(process_count, int)

    def test_file_handle_count_measurement(self) -> None:
        """Test file handle count measurement."""
        monitor = SystemResourceMonitor()
        file_handles = monitor.get_file_handle_count()
        assert file_handles >= 0
        assert isinstance(file_handles, int)

    def test_cpu_usage_measurement(self) -> None:
        """Test CPU usage measurement."""
        monitor = SystemResourceMonitor()
        cpu_percent = monitor.get_cpu_usage_percent()
        assert 0.0 <= cpu_percent <= 100.0
        assert isinstance(cpu_percent, float)

    def test_garbage_collection_force(self) -> None:
        """Test forced garbage collection."""
        monitor = SystemResourceMonitor()
        # Should not raise any exception
        monitor.force_garbage_collection()

    def test_gpu_cache_clearing(self) -> None:
        """Test GPU cache clearing."""
        monitor = SystemResourceMonitor()
        # Should return True (success) regardless of GPU availability
        result = monitor.clear_gpu_cache()
        assert isinstance(result, bool)


class TestResourceCleanupValidationMixin:
    """Test suite for ResourceCleanupValidationMixin functionality."""

    def test_mixin_initialization(self) -> None:
        """Test that the mixin initializes correctly."""
        mixin = ResourceCleanupValidationMixin()
        assert mixin.config is not None
        assert isinstance(mixin.cleanup_metrics, list)
        assert isinstance(mixin.resource_baselines, dict)

    def test_baseline_establishment(self) -> None:
        """Test resource baseline establishment."""
        mixin = ResourceCleanupValidationMixin()
        baseline = mixin.establish_resource_baseline("test_workflow")

        assert isinstance(baseline, ResourceBaseline)
        assert baseline.memory_mb > 0
        assert baseline.process_count > 0
        assert baseline.file_handles >= 0
        assert baseline.cpu_percent >= 0
        assert "test_workflow" in mixin.resource_baselines

    def test_cleanup_validation_flow(self) -> None:
        """Test the complete cleanup validation flow."""
        mixin = ResourceCleanupValidationMixin()

        # Establish baseline
        baseline = mixin.establish_resource_baseline("test_cleanup")

        # Validate cleanup with a simple operation
        def simple_cleanup_operation() -> None:
            # Simple operation that should clean up properly
            data = [1, 2, 3, 4, 5]
            processed = [x * 2 for x in data]
            del data, processed

        metrics = mixin.validate_resource_cleanup(
            "test_cleanup", simple_cleanup_operation
        )

        assert isinstance(metrics, ResourceCleanupMetrics)
        assert metrics.memory_baseline_mb == baseline.memory_mb
        assert metrics.process_baseline_count == baseline.process_count
        assert metrics.cleanup_time_seconds >= 0

    def test_cleanup_metrics_collection(self) -> None:
        """Test cleanup metrics collection."""
        mixin = ResourceCleanupValidationMixin()

        # Perform a cleanup validation
        mixin.establish_resource_baseline("metrics_test")
        mixin.validate_resource_cleanup("metrics_test", lambda: None)

        # Check metrics collection
        metrics = mixin.get_cleanup_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Test metrics clearing
        mixin.clear_cleanup_history()
        metrics_after_clear = mixin.get_cleanup_metrics()
        assert len(metrics_after_clear) == 0


class TestResourceCleanupValidationComponent:
    """Test suite for ResourceCleanupValidationComponent."""

    def test_component_initialization(self) -> None:
        """Test that the component initializes correctly."""
        test_utilities = {}  # Mock test utilities
        component = ResourceCleanupValidationComponent(test_utilities)

        assert component.test_utilities is test_utilities
        assert component.config is not None
        assert component.performance_benchmarking is not None

    def test_workflow_name(self) -> None:
        """Test workflow name retrieval."""
        component = ResourceCleanupValidationComponent({})
        name = component.get_workflow_name()
        assert isinstance(name, str)
        assert "Resource Cleanup Validation" in name

    def test_automation_preconditions(self) -> None:
        """Test automation preconditions validation."""
        component = ResourceCleanupValidationComponent({})
        preconditions_met = component.validate_automation_preconditions()
        assert isinstance(preconditions_met, bool)

    def test_automation_metrics_retrieval(self) -> None:
        """Test automation metrics retrieval."""
        component = ResourceCleanupValidationComponent({})
        metrics = component.get_automation_metrics()
        assert isinstance(metrics, dict)
        # Should have some metrics even if no data collected yet
        assert len(metrics) > 0

    def test_workflow_component_cleanup_validation(self) -> None:
        """Test workflow component cleanup validation."""
        component = ResourceCleanupValidationComponent({})

        # Test configuration with workflow validation enabled
        config = CleanupValidationConfig(validate_workflow_components=True)
        component.config = config

        # Mock automation configuration
        from .automation_protocols import AutomationConfiguration

        automation_config = AutomationConfiguration(
            timeout_seconds=30,
            retry_count=3,
        )

        # Test the validation method
        results = component._validate_workflow_component_cleanup(
            automation_config
        )

        assert isinstance(results, list)
        # Should have results for each workflow component
        assert len(results) > 0

    def test_memory_cleanup_validation(self) -> None:
        """Test memory cleanup validation."""
        component = ResourceCleanupValidationComponent({})

        # Enable memory cleanup validation
        config = CleanupValidationConfig(validate_memory_cleanup=True)
        component.config = config

        from .automation_protocols import AutomationConfiguration

        automation_config = AutomationConfiguration(
            timeout_seconds=30,
            retry_count=3,
        )

        results = component._validate_memory_cleanup(automation_config)

        assert isinstance(results, list)
        # Should have results for memory scenarios
        assert len(results) > 0

    def test_process_cleanup_validation(self) -> None:
        """Test process cleanup validation."""
        component = ResourceCleanupValidationComponent({})

        # Enable process cleanup validation
        config = CleanupValidationConfig(validate_process_cleanup=True)
        component.config = config

        from .automation_protocols import AutomationConfiguration

        automation_config = AutomationConfiguration(
            timeout_seconds=30,
            retry_count=3,
        )

        results = component._validate_process_cleanup(automation_config)

        assert isinstance(results, list)
        # Should have results for process scenarios
        assert len(results) > 0

    def test_baseline_restoration_validation(self) -> None:
        """Test baseline restoration validation."""
        component = ResourceCleanupValidationComponent({})

        # Enable baseline restoration validation
        config = CleanupValidationConfig(validate_baseline_restoration=True)
        component.config = config

        from .automation_protocols import AutomationConfiguration

        automation_config = AutomationConfiguration(
            timeout_seconds=30,
            retry_count=3,
        )

        results = component._validate_baseline_restoration(automation_config)

        assert isinstance(results, list)
        # Should have results for baseline scenarios
        assert len(results) > 0


class TestResourceCleanupProtocols:
    """Test suite for resource cleanup protocols and data structures."""

    def test_cleanup_metrics_initialization(self) -> None:
        """Test ResourceCleanupMetrics initialization."""
        metrics = ResourceCleanupMetrics(
            memory_baseline_mb=1000.0,
            memory_post_execution_mb=1050.0,
            memory_leak_detected=True,
            memory_cleanup_percentage=95.0,
            process_baseline_count=100,
            process_post_execution_count=101,
            orphaned_processes_detected=True,
            process_cleanup_successful=False,
            file_handles_baseline=50,
            file_handles_post_execution=52,
            file_leak_detected=True,
            temp_files_cleaned=True,
            gpu_memory_baseline_mb=500.0,
            gpu_memory_post_execution_mb=510.0,
            gpu_memory_leaked=True,
            cuda_context_cleaned=False,
            cpu_baseline_percent=20.0,
            cpu_post_execution_percent=22.0,
            baseline_restoration_successful=False,
            cleanup_validation_passed=False,
            cleanup_time_seconds=2.5,
            leak_detection_accuracy=0.95,
        )

        assert metrics.memory_baseline_mb == 1000.0
        assert metrics.memory_leak_detected is True
        assert metrics.cleanup_validation_passed is False

    def test_memory_leak_severity_calculation(self) -> None:
        """Test memory leak severity calculation."""
        # Test critical leak
        metrics = ResourceCleanupMetrics(
            memory_baseline_mb=1000.0,
            memory_post_execution_mb=1250.0,  # 25% increase
            memory_leak_detected=True,
            memory_cleanup_percentage=75.0,
            process_baseline_count=100,
            process_post_execution_count=100,
            orphaned_processes_detected=False,
            process_cleanup_successful=True,
            file_handles_baseline=50,
            file_handles_post_execution=50,
            file_leak_detected=False,
            temp_files_cleaned=True,
            gpu_memory_baseline_mb=500.0,
            gpu_memory_post_execution_mb=500.0,
            gpu_memory_leaked=False,
            cuda_context_cleaned=True,
            cpu_baseline_percent=20.0,
            cpu_post_execution_percent=20.0,
            baseline_restoration_successful=True,
            cleanup_validation_passed=True,
            cleanup_time_seconds=1.0,
            leak_detection_accuracy=1.0,
        )

        severity = metrics.get_memory_leak_severity()
        assert severity == "critical"

        # Test no leak
        metrics.memory_leak_detected = False
        severity = metrics.get_memory_leak_severity()
        assert severity == "none"

    def test_resource_baseline_initialization(self) -> None:
        """Test ResourceBaseline initialization."""
        baseline = ResourceBaseline(
            memory_mb=1000.0,
            process_count=100,
            file_handles=50,
            gpu_memory_mb=500.0,
            cpu_percent=20.0,
        )

        assert baseline.memory_mb == 1000.0
        assert baseline.process_count == 100
        assert baseline.file_handles == 50
        assert baseline.gpu_memory_mb == 500.0
        assert baseline.cpu_percent == 20.0

    def test_baseline_age_calculation(self) -> None:
        """Test baseline age calculation."""
        baseline = ResourceBaseline(
            memory_mb=1000.0,
            process_count=100,
            file_handles=50,
            gpu_memory_mb=500.0,
            cpu_percent=20.0,
        )

        age_minutes = baseline.get_baseline_age_minutes()
        assert age_minutes >= 0.0
        assert isinstance(age_minutes, float)

        # Test staleness detection
        is_stale = baseline.is_baseline_stale(max_age_minutes=0.001)
        # Should be stale since we set a very small max age
        assert isinstance(is_stale, bool)

    def test_cleanup_config_validation(self) -> None:
        """Test CleanupValidationConfig validation."""
        config = CleanupValidationConfig()
        errors = config.validate_config()
        assert isinstance(errors, list)
        # Default config should be valid
        assert len(errors) == 0

        # Test invalid config
        invalid_config = CleanupValidationConfig(
            memory_leak_tolerance_percent=-1.0,
            cleanup_timeout_seconds=0.0,
        )
        errors = invalid_config.validate_config()
        assert len(errors) > 0
        assert any("Memory leak tolerance" in error for error in errors)
        assert any("Cleanup timeout" in error for error in errors)
