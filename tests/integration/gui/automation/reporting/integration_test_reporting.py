"""
Main orchestrator for comprehensive integration test reporting. This
module serves as the main entry point for the comprehensive
integration test reporting system, coordinating specialized modules
for data aggregation, stakeholder reporting, trend analysis, and
multi-format export.
"""

from datetime import datetime
from typing import Any

from ..automation_orchestrator import AutomationReporterImpl
from ..automation_protocols import AutomationConfiguration, AutomationResult
from ..performance_benchmarking import PerformanceBenchmarkingComponent
from ..resource_cleanup_validation import ResourceCleanupValidationComponent
from .analysis_engine import AnalysisEngine
from .data_aggregation import TestDataAggregator
from .export_manager import MultiFormatExportManager
from .metrics_compiler import ReportingMetricsCompiler
from .stakeholder_reporting import (
    StakeholderReportConfig,
    StakeholderReportGenerator,
)
from .validation_utils import ReportingValidationUtils


class IntegrationTestReportingComponent:
    """
    Main orchestrator for comprehensive integration test reporting. This
    component coordinates all specialized reporting modules to provide
    comprehensive stakeholder-specific reporting with data aggregation,
    trend analysis, and multi-format export capabilities.
    """

    def __init__(
        self,
        test_utilities: Any,
        config: StakeholderReportConfig | None = None,
    ) -> None:
        """
        Initialize comprehensive integration test reporting component. Args:
        test_utilities: Test utilities for integration config: Stakeholder
        report configuration
        """
        self.test_utilities = test_utilities
        self.config = config or StakeholderReportConfig()

        # Initialize infrastructure components
        self.automation_reporter = AutomationReporterImpl()
        self.performance_component = PerformanceBenchmarkingComponent(
            test_utilities
        )
        self.resource_cleanup_component = ResourceCleanupValidationComponent(
            test_utilities
        )

        # Initialize specialized reporting modules
        self.data_aggregator = TestDataAggregator(
            self.automation_reporter,
            self.performance_component,
            self.resource_cleanup_component,
        )
        self.stakeholder_generator = StakeholderReportGenerator(self.config)
        self.export_manager = MultiFormatExportManager()

        # Historical data for trend analysis
        self.historical_data: list[dict[str, Any]] = []

        # Initialize utility modules
        self.metrics_compiler = ReportingMetricsCompiler(
            self.config, self.historical_data
        )

    def get_workflow_name(self) -> str:
        """Get the name of this comprehensive reporting workflow."""
        return "CrackSeg Comprehensive Integration Test Reporting Suite"

    def execute_automated_workflow(
        self, automation_config: dict[str, Any]
    ) -> AutomationResult:
        """
        Execute comprehensive integration test reporting workflow. Args:
        automation_config: Automation configuration Returns: AutomationResult
        with comprehensive reporting results
        """
        AutomationConfiguration(**automation_config)
        start_time = datetime.now()

        try:
            # Step 1: Aggregate data from all testing phases (9.1-9.7)
            aggregated_data = (
                self.data_aggregator.aggregate_comprehensive_data()
            )

            # Store historical data for trend analysis
            self.historical_data.append(aggregated_data)

            # Step 2: Generate stakeholder-specific reports
            stakeholder_reports = (
                self.stakeholder_generator.generate_all_reports(
                    aggregated_data
                )
            )

            # Step 3: Perform trend analysis and regression detection
            analysis_results = self._perform_comprehensive_analysis(
                aggregated_data
            )

            # Step 4: Export reports in multiple formats
            export_artifacts = self.export_manager.export_stakeholder_reports(
                stakeholder_reports,
                analysis_results,
                self.config.export_formats,
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Calculate success metrics
            total_reports = len(stakeholder_reports)
            successful_reports = sum(
                1
                for report in stakeholder_reports.values()
                if report.get("success", False)
            )

            return AutomationResult(
                workflow_name=self.get_workflow_name(),
                success=successful_reports == total_reports,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                test_count=total_reports,
                passed_count=successful_reports,
                failed_count=total_reports - successful_reports,
                error_details=ReportingValidationUtils.extract_reporting_errors(
                    stakeholder_reports
                ),
                performance_metrics=self.metrics_compiler.compile_reporting_metrics(
                    aggregated_data
                ),
                artifacts_generated=export_artifacts,
                metadata={
                    "reporting_type": "comprehensive_stakeholder_reporting",
                    "stakeholder_coverage": list(stakeholder_reports.keys()),
                    "data_sources": [
                        "9.1",
                        "9.2",
                        "9.3",
                        "9.4",
                        "9.5",
                        "9.6",
                        "9.7",
                    ],
                    "analysis_features": [
                        "trend_analysis",
                        "regression_detection",
                        "multi_format_export",
                    ],
                    "export_formats": self.config.export_formats,
                    "rtx_3070_ti_optimization": "enabled",
                },
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return AutomationResult(
                workflow_name=self.get_workflow_name(),
                success=False,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                error_details=[f"Reporting workflow failed: {str(e)}"],
                performance_metrics={},
                artifacts_generated=[],
                metadata={
                    "reporting_type": "comprehensive_stakeholder_reporting",
                    "error": str(e),
                },
            )

    def validate_automation_preconditions(self) -> bool:
        """
        Validate that comprehensive reporting preconditions are met. Returns:
        True if all preconditions are satisfied
        """
        try:
            # Validate infrastructure components
            perf_valid = (
                self.performance_component.validate_automation_preconditions()
            )
            cleanup_valid = (
                self.resource_cleanup_component.validate_automation_preconditions()
            )
            components_valid = perf_valid and cleanup_valid

            # Validate configuration
            config_valid = (
                ReportingValidationUtils.validate_reporting_configuration(
                    self.config
                )
            )

            # Validate output directories
            output_dirs_valid = (
                ReportingValidationUtils.validate_output_directories()
            )

            return components_valid and config_valid and output_dirs_valid

        except Exception:
            return False

    def get_automation_metrics(self) -> dict[str, float]:
        """
        Get comprehensive reporting automation metrics. Returns: Dictionary of
        automation metrics
        """
        return self.metrics_compiler.get_automation_metrics()

    def _perform_comprehensive_analysis(
        self, aggregated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Perform comprehensive analysis including trends and regression
        detection. Args: aggregated_data: Aggregated testing data Returns:
        Analysis results
        """
        analysis_results = {}

        # Trend analysis
        if self.config.include_trends and len(self.historical_data) > 1:
            analysis_engine = AnalysisEngine(self.historical_data)
            analysis_results["trend_analysis"] = {
                "performance_trends": (
                    analysis_engine.trend_engine.analyze_performance_trends()
                ),
                "quality_trends": (
                    analysis_engine.trend_engine.analyze_quality_trends()
                ),
                "predictions": (
                    analysis_engine.trend_engine.predict_future_trends()
                ),
            }

        # Regression detection
        if (
            self.config.include_regression_analysis
            and len(self.historical_data) > 1
        ):
            analysis_engine = AnalysisEngine(self.historical_data)
            analysis_results["regression_detection"] = (
                analysis_engine.regression_engine.generate_regression_report()
            )

        # Cross-phase metrics (placeholder - needs implementation)
        # cross_phase_metrics = (
        #     self.data_aggregator.calculate_cross_phase_metrics(
        #         aggregated_data
        #     )
        # )
        # analysis_results["cross_phase_metrics"] = cross_phase_metrics

        return analysis_results

    def get_reporting_status(self) -> dict[str, Any]:
        """
        Get current reporting system status. Returns: Current reporting system
        status
        """
        return self.metrics_compiler.get_reporting_status()
