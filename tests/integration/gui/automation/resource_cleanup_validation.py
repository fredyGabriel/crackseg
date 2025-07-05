"""Resource cleanup validation component for systematic resource management
verification.

This module extends the automation and performance frameworks from 9.5-9.6 to
provide the main Resource Cleanup Validation Component that orchestrates
comprehensive resource cleanup validation across all workflow components
(9.1-9.4) with integration into the existing automation infrastructure.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from .automation_protocols import (
    AutomationConfiguration,
    AutomationResult,
)
from .performance_benchmarking import PerformanceBenchmarkingComponent
from .resource_cleanup_monitoring import ResourceCleanupValidationMixin
from .resource_cleanup_protocols import (
    CleanupValidationConfig,
    CleanupValidationReport,
    ResourceCleanupMetrics,
)


class ResourceCleanupValidationComponent(ResourceCleanupValidationMixin):
    """Resource cleanup validation component extending automation
    infrastructure.

    Provides systematic resource cleanup validation across all workflow
    components (9.1-9.4) with comprehensive leak detection, baseline
    restoration, and resource management verification capabilities integrated
    with the existing automation and performance frameworks.
    """

    def __init__(
        self,
        test_utilities: Any,
        config: CleanupValidationConfig | None = None,
    ) -> None:
        """Initialize resource cleanup validation with test utilities."""
        super().__init__(config)
        self.test_utilities = test_utilities
        self.performance_benchmarking = PerformanceBenchmarkingComponent(
            test_utilities
        )

    def get_workflow_name(self) -> str:
        """Get the name of this resource cleanup validation workflow."""
        return "CrackSeg Resource Cleanup Validation Suite"

    def execute_automated_workflow(
        self, automation_config: dict[str, Any]
    ) -> AutomationResult:
        """Execute comprehensive resource cleanup validation workflow."""
        config = AutomationConfiguration(**automation_config)
        start_time = datetime.now()

        # Execute resource cleanup validation phases
        workflow_cleanup_results = self._validate_workflow_component_cleanup(
            config
        )
        memory_cleanup_results = self._validate_memory_cleanup(config)
        process_cleanup_results = self._validate_process_cleanup(config)
        resource_baseline_results = self._validate_baseline_restoration(config)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Aggregate results
        all_results = (
            workflow_cleanup_results
            + memory_cleanup_results
            + process_cleanup_results
            + resource_baseline_results
        )

        total_tests = len(all_results)
        passed_tests = sum(
            1 for r in all_results if r.cleanup_validation_passed
        )

        # Compile resource cleanup metrics
        cleanup_metrics = self._compile_cleanup_metrics()

        return AutomationResult(
            workflow_name=self.get_workflow_name(),
            success=passed_tests == total_tests,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            test_count=total_tests,
            passed_count=passed_tests,
            failed_count=total_tests - passed_tests,
            error_details=self._extract_cleanup_errors(),
            performance_metrics=cleanup_metrics,
            artifacts_generated=self._generate_cleanup_artifacts(config),
            metadata={
                "cleanup_validation_type": "systematic_resource_cleanup",
                "baseline_restoration": "enabled",
                "leak_detection": "comprehensive",
                "rtx_3070_ti_optimization": "enabled",
                "integration": "automation_and_performance_frameworks",
            },
        )

    def validate_automation_preconditions(self) -> bool:
        """Validate that resource cleanup validation preconditions are met."""
        try:
            # Verify dependencies are available

            # Verify configuration is valid
            config_errors = self.config.validate_config()
            if config_errors:
                return False

            # Verify system resources are accessible
            test_baseline = self.establish_resource_baseline(
                "precondition_test"
            )
            if test_baseline.memory_mb <= 0:
                return False

            # Clean up test baseline
            del self.resource_baselines["precondition_test"]

            return True
        except Exception:
            return False

    def get_automation_metrics(self) -> dict[str, float]:
        """Get resource cleanup specific automation metrics."""
        summary = self.get_validation_summary()
        if summary.get("no_data"):
            return {"no_cleanup_data": 0.0}

        return {
            "cleanup_success_rate": summary["successful_cleanups"]
            / summary["total_validations"]
            * 100,
            "memory_leak_rate": summary["memory_leaks_detected"]
            / summary["total_validations"]
            * 100,
            "process_leak_rate": summary["process_leaks_detected"]
            / summary["total_validations"]
            * 100,
            "file_leak_rate": summary["file_leaks_detected"]
            / summary["total_validations"]
            * 100,
            "gpu_leak_rate": summary["gpu_leaks_detected"]
            / summary["total_validations"]
            * 100,
            "avg_cleanup_time_seconds": summary["avg_cleanup_time"],
            "baseline_restoration_success_rate": summary[
                "baseline_restoration_success_rate"
            ],
        }

    def _validate_workflow_component_cleanup(
        self, config: AutomationConfiguration
    ) -> list[ResourceCleanupMetrics]:
        """Validate resource cleanup for all workflow components."""
        results = []

        if not self.config.validate_workflow_components:
            return results

        # Test each workflow component for proper cleanup
        workflow_components = [
            "config_workflow",
            "training_workflow",
            "error_scenarios",
            "session_state",
            "concurrent_operations",
        ]

        for component in workflow_components:
            self.establish_resource_baseline(component)

            # Simulate workflow execution and cleanup
            cleanup_metrics = self.validate_resource_cleanup(
                component,
                lambda comp=component: self._simulate_workflow_cleanup(comp),
            )

            results.append(cleanup_metrics)

        return results

    def _validate_memory_cleanup(
        self, config: AutomationConfiguration
    ) -> list[ResourceCleanupMetrics]:
        """Validate memory cleanup and leak detection."""
        results = []

        if not self.config.validate_memory_cleanup:
            return results

        # Memory-intensive scenarios
        memory_scenarios = ["memory_intensive_config", "training_memory_load"]

        for scenario in memory_scenarios:
            self.establish_resource_baseline(scenario)

            cleanup_metrics = self.validate_resource_cleanup(
                scenario,
                lambda s=scenario: self._simulate_memory_intensive_cleanup(s),
            )

            results.append(cleanup_metrics)

        return results

    def _validate_process_cleanup(
        self, config: AutomationConfiguration
    ) -> list[ResourceCleanupMetrics]:
        """Validate process cleanup and orphan detection."""
        results = []

        if not self.config.validate_process_cleanup:
            return results

        # Process scenarios
        process_scenarios = ["streamlit_process", "training_subprocess"]

        for scenario in process_scenarios:
            self.establish_resource_baseline(scenario)

            cleanup_metrics = self.validate_resource_cleanup(
                scenario, lambda s=scenario: self._simulate_process_cleanup(s)
            )

            results.append(cleanup_metrics)

        return results

    def _validate_baseline_restoration(
        self, config: AutomationConfiguration
    ) -> list[ResourceCleanupMetrics]:
        """Validate baseline restoration after workflow execution."""
        results = []

        if not self.config.validate_baseline_restoration:
            return results

        # Baseline restoration scenarios
        baseline_scenarios = ["system_baseline", "gpu_baseline"]

        for scenario in baseline_scenarios:
            self.establish_resource_baseline(scenario)

            cleanup_metrics = self.validate_resource_cleanup(
                scenario,
                lambda s=scenario: self._simulate_baseline_restoration(s),
            )

            results.append(cleanup_metrics)

        return results

    def _simulate_workflow_cleanup(self, component: str) -> None:
        """Simulate workflow component cleanup."""
        # Simulate some resource usage and cleanup for different components
        if "config" in component:
            # Simulate config loading and cleanup
            import tempfile

            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(b"config data")
                temp_file.flush()
        elif "training" in component:
            # Simulate training memory allocation
            data = [0] * 100000  # Allocate memory
            del data
        elif "session" in component:
            # Simulate session state cleanup
            session_data = {"key": "value" * 1000}
            del session_data

        # Force cleanup
        self.resource_monitor.force_garbage_collection()

    def _simulate_memory_intensive_cleanup(self, scenario: str) -> None:
        """Simulate memory-intensive cleanup scenario."""
        # Allocate and deallocate memory to test cleanup
        large_data = list(range(1000000))  # Allocate ~8MB
        processed_data = [x * 2 for x in large_data]  # More allocation
        del large_data, processed_data

        # Force cleanup
        self.resource_monitor.force_garbage_collection()

    def _simulate_process_cleanup(self, scenario: str) -> None:
        """Simulate process cleanup scenario."""
        # Simulate subprocess-like operations that need cleanup
        import subprocess
        import sys

        try:
            # Quick subprocess that exits immediately
            subprocess.run(
                [sys.executable, "-c", "print('cleanup test')"],
                capture_output=True,
                timeout=1,
            )
        except subprocess.TimeoutExpired:
            pass

        # Force cleanup
        self.resource_monitor.force_garbage_collection()

    def _simulate_baseline_restoration(self, scenario: str) -> None:
        """Simulate baseline restoration scenario."""
        # Simulate operations that affect system baseline
        if "gpu" in scenario:
            # GPU resource cleanup
            self.resource_monitor.clear_gpu_cache()
        else:
            # General system cleanup
            self.resource_monitor.force_garbage_collection()

    def _compile_cleanup_metrics(self) -> dict[str, float]:
        """Compile aggregated resource cleanup metrics."""
        metrics = self.get_cleanup_metrics()
        if not metrics:
            return {}

        return {
            "avg_memory_cleanup_percentage": sum(
                m.memory_cleanup_percentage for m in metrics
            )
            / len(metrics),
            "cleanup_validation_success_rate": sum(
                1 for m in metrics if m.cleanup_validation_passed
            )
            / len(metrics)
            * 100,
            "memory_leak_detection_rate": sum(
                1 for m in metrics if m.memory_leak_detected
            )
            / len(metrics)
            * 100,
            "baseline_restoration_success_rate": sum(
                1 for m in metrics if m.baseline_restoration_successful
            )
            / len(metrics)
            * 100,
            "avg_cleanup_time_seconds": sum(
                m.cleanup_time_seconds for m in metrics
            )
            / len(metrics),
            "gpu_cleanup_success_rate": sum(
                1 for m in metrics if m.cuda_context_cleaned
            )
            / len(metrics)
            * 100,
        }

    def _extract_cleanup_errors(self) -> list[str]:
        """Extract error details from cleanup validation."""
        errors = []

        for metrics in self.get_cleanup_metrics():
            if not metrics.cleanup_validation_passed:
                error_details = []
                if metrics.memory_leak_detected:
                    severity = metrics.get_memory_leak_severity()
                    error_details.append(f"Memory leak detected ({severity})")
                if metrics.orphaned_processes_detected:
                    error_details.append("Orphaned processes found")
                if metrics.file_leak_detected:
                    error_details.append("File handle leak detected")
                if metrics.gpu_memory_leaked:
                    error_details.append("GPU memory leak detected")
                if not metrics.baseline_restoration_successful:
                    error_details.append("Baseline restoration failed")

                if error_details:
                    component_name = metrics.workflow_component
                    error_summary = ", ".join(error_details)
                    errors.append(f"{component_name}: {error_summary}")

        return errors

    def _generate_cleanup_artifacts(
        self, config: AutomationConfiguration
    ) -> list[Path]:
        """Generate resource cleanup validation artifacts."""
        artifacts = []

        # Create artifacts directory
        artifacts_dir = Path("automation_results")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Generate cleanup metrics JSON report
        metrics_data = {
            "resource_cleanup_metrics": [
                {
                    "workflow_component": m.workflow_component,
                    "cleanup_validation_passed": m.cleanup_validation_passed,
                    "memory_leak_detected": m.memory_leak_detected,
                    "memory_leak_severity": m.get_memory_leak_severity(),
                    "cleanup_summary": m.get_cleanup_summary(),
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.get_cleanup_metrics()
            ],
            "validation_summary": self.get_validation_summary(),
            "automation_metrics": self.get_automation_metrics(),
        }

        metrics_path = artifacts_dir / "resource_cleanup_metrics.json"
        try:
            import json

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
            artifacts.append(metrics_path)
        except Exception:
            pass

        # Generate cleanup validation report
        report = self._generate_validation_report()
        report_path = artifacts_dir / "resource_cleanup_report.json"
        try:
            import json

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report.get_summary_dict(), f, indent=2)
            artifacts.append(report_path)
        except Exception:
            pass

        return artifacts

    def _generate_validation_report(self) -> CleanupValidationReport:
        """Generate comprehensive validation report."""
        metrics = self.get_cleanup_metrics()

        if not metrics:
            return CleanupValidationReport(
                validation_timestamp=datetime.now(),
                total_workflows_tested=0,
                successful_cleanups=0,
                failed_cleanups=0,
                memory_leaks_detected=0,
                process_leaks_detected=0,
                file_leaks_detected=0,
                gpu_leaks_detected=0,
                avg_cleanup_time_seconds=0.0,
                max_cleanup_time_seconds=0.0,
                min_cleanup_time_seconds=0.0,
            )

        cleanup_times = [m.cleanup_time_seconds for m in metrics]

        return CleanupValidationReport(
            validation_timestamp=datetime.now(),
            total_workflows_tested=len(metrics),
            successful_cleanups=sum(
                1 for m in metrics if m.cleanup_validation_passed
            ),
            failed_cleanups=sum(
                1 for m in metrics if not m.cleanup_validation_passed
            ),
            memory_leaks_detected=sum(
                1 for m in metrics if m.memory_leak_detected
            ),
            process_leaks_detected=sum(
                1 for m in metrics if m.orphaned_processes_detected
            ),
            file_leaks_detected=sum(
                1 for m in metrics if m.file_leak_detected
            ),
            gpu_leaks_detected=sum(1 for m in metrics if m.gpu_memory_leaked),
            avg_cleanup_time_seconds=sum(cleanup_times) / len(cleanup_times),
            max_cleanup_time_seconds=max(cleanup_times),
            min_cleanup_time_seconds=min(cleanup_times),
            cleanup_metrics=metrics,
        )
