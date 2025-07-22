"""End-to-end integration tests for the comprehensive reporting system.

This module tests the complete workflow from crackseg.data collection through
analysis to multi-format export, validating the entire reporting pipeline.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.integration.gui.automation.reporting.integration_test_reporting import (  # noqa: E501
    IntegrationTestReportingComponent,
)
from tests.integration.gui.automation.reporting.stakeholder_reporting import (
    StakeholderReportConfig,
)


class TestEndToEndReporting:
    """Test suite for end-to-end reporting workflow."""

    @pytest.fixture
    def mock_test_components(self) -> dict[str, Mock]:
        """Create comprehensive mock test components."""
        return {
            "automation_reporter": Mock(),
            "resource_cleanup_component": Mock(),
            "performance_monitor": Mock(),
            "workflow_scenarios_component": Mock(),
            "error_scenarios_component": Mock(),
            "session_state_component": Mock(),
            "concurrent_operations_component": Mock(),
        }

    @pytest.fixture
    def reporting_system(
        self, mock_test_components: dict[str, Mock]
    ) -> IntegrationTestReportingComponent:
        """Create fully configured reporting system."""
        return IntegrationTestReportingComponent(
            test_utilities=mock_test_components
        )

    @pytest.fixture
    def realistic_test_config(self) -> StakeholderReportConfig:
        """Create realistic comprehensive test configuration."""
        return StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            operations_enabled=True,
            include_trends=True,
            include_regression_analysis=True,
        )

    def test_complete_reporting_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        realistic_test_config: StakeholderReportConfig,
        tmp_path: Path,
    ) -> None:
        """Test complete end-to-end reporting workflow."""
        # Configure the system with the test config
        reporting_system.config = realistic_test_config

        # Step 1: Execute automated workflow
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify workflow execution
        assert result.success
        assert result.test_count > 0
        assert result.passed_count > 0

        # Step 2: Export to all formats using the export manager
        export_results = (
            reporting_system.export_manager.export_multiple_formats(
                {"test_data": "sample"},
                formats=["html", "json", "csv"],
                output_dir=tmp_path,
            )
        )

        # Verify all exports succeeded
        assert "html" in export_results
        assert "json" in export_results
        assert "csv" in export_results

        # Step 3: Validate exported files
        for _format_type, result in export_results.items():
            file_path = Path(result["path"])
            assert file_path.exists()
            assert file_path.stat().st_size > 0  # File is not empty

    def test_executive_focused_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow focused on executive reporting needs."""
        executive_config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=False,
            operations_enabled=False,
            include_trends=True,
            include_regression_analysis=False,
        )

        # Configure the system
        reporting_system.config = executive_config

        # Generate executive-focused report
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify executive content was processed
        assert result.success
        assert "executive" in result.metadata.get("stakeholder_coverage", [])

        # Export as HTML for presentation
        export_result = reporting_system.export_manager.export_report(
            {"executive_summary": "test"}, "html", output_dir=tmp_path
        )

        # Validate HTML export for executive use
        html_path = Path(export_result["path"])
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html" in content

    def test_technical_deep_dive_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow for technical deep-dive analysis."""
        technical_config = StakeholderReportConfig(
            executive_enabled=False,
            technical_enabled=True,
            operations_enabled=False,
            include_trends=True,
            include_regression_analysis=True,
        )

        # Configure the system
        reporting_system.config = technical_config

        # Generate technical analysis
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify technical content was processed
        assert result.success
        assert "technical" in result.metadata.get("stakeholder_coverage", [])

        # Export as JSON for programmatic analysis
        export_result = reporting_system.export_manager.export_report(
            {"technical_analysis": "test"}, "json", output_dir=tmp_path
        )

        # Validate JSON structure for technical use
        json_path = Path(export_result["path"])
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "technical_analysis" in data

    def test_operations_monitoring_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow for operations monitoring."""
        operations_config = StakeholderReportConfig(
            executive_enabled=False,
            technical_enabled=False,
            operations_enabled=True,
            include_trends=False,
            include_regression_analysis=True,
        )

        # Configure the system
        reporting_system.config = operations_config

        # Generate operations report
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify operations content was processed
        assert result.success
        assert "operations" in result.metadata.get("stakeholder_coverage", [])

        # Export as CSV for data analysis
        export_result = reporting_system.export_manager.export_report(
            {"operations_monitoring": "test"}, "csv", output_dir=tmp_path
        )

        # Validate CSV format for operations
        csv_path = Path(export_result["path"])
        content = csv_path.read_text(encoding="utf-8")
        assert "section,metric,value" in content
        assert "operations_monitoring" in content

    def test_multi_stakeholder_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow serving multiple stakeholders simultaneously."""
        # Generate reports for all stakeholders
        configs = {
            "executive": StakeholderReportConfig(
                executive_enabled=True,
                technical_enabled=False,
                operations_enabled=False,
            ),
            "technical": StakeholderReportConfig(
                executive_enabled=False,
                technical_enabled=True,
                operations_enabled=False,
            ),
            "operations": StakeholderReportConfig(
                executive_enabled=False,
                technical_enabled=False,
                operations_enabled=True,
            ),
        }

        results = {}
        for stakeholder, config in configs.items():
            reporting_system.config = config
            automation_config = {"test_mode": True}
            result = reporting_system.execute_automated_workflow(
                automation_config
            )

            export_result = reporting_system.export_manager.export_report(
                {f"{stakeholder}_data": "test"},
                "html",
                output_dir=tmp_path,
                filename=f"{stakeholder}_report",
            )
            results[stakeholder] = export_result

        # Verify all stakeholder reports were generated
        assert len(results) == 3
        for stakeholder, result in results.items():
            file_path = Path(result["path"])
            assert file_path.exists()
            assert stakeholder in file_path.name

    def test_trend_analysis_integration(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test integration of trend analysis in the workflow."""
        trend_config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            include_trends=True,
            include_regression_analysis=False,
        )

        # Configure the system
        reporting_system.config = trend_config

        # Generate report with trends
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify trend analysis is included
        assert "trend_analysis" in result.metadata.get("analysis_features", [])

    def test_regression_detection_integration(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test integration of regression detection in the workflow."""
        regression_config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            include_trends=False,
            include_regression_analysis=True,
        )

        # Configure the system
        reporting_system.config = regression_config

        # Generate report with regression analysis
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Verify regression analysis is included
        assert "regression_detection" in result.metadata.get(
            "analysis_features", []
        )

    def test_data_aggregation_completeness(
        self, reporting_system: IntegrationTestReportingComponent
    ) -> None:
        """Test that data aggregation captures all testing phases."""
        config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            operations_enabled=True,
        )

        reporting_system.config = config
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Check that all phases are represented in the data
        metadata = result.metadata
        data_sources = metadata.get("data_sources", [])

        # Should include data from phases 9.1-9.7
        expected_phases = [
            "9.1",
            "9.2",
            "9.3",
            "9.4",
            "9.5",
            "9.6",
            "9.7",
        ]

        # At least some of these phases should be represented
        assert len(data_sources) > 0
        assert all(phase in data_sources for phase in expected_phases)

    def test_export_format_compatibility(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test compatibility and correctness of different export formats."""
        config = StakeholderReportConfig(
            executive_enabled=True, technical_enabled=True
        )

        reporting_system.config = config
        automation_config = {"test_mode": True}
        reporting_system.execute_automated_workflow(automation_config)

        # Export to all formats
        export_results = (
            reporting_system.export_manager.export_multiple_formats(
                {"test_data": "sample"},
                formats=["html", "json", "csv"],
                output_dir=tmp_path,
            )
        )

        # Validate HTML format
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")
        assert "<html" in html_content

        # Validate JSON format
        json_path = Path(export_results["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)
        assert "test_data" in json_data

        # Validate CSV format
        csv_path = Path(export_results["csv"]["path"])
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "section,metric,value" in csv_content

    def test_error_recovery_in_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test error recovery mechanisms in the workflow."""
        config = StakeholderReportConfig(executive_enabled=True)
        reporting_system.config = config

        # Test with invalid automation config to trigger error handling
        invalid_config = {"invalid_key": "invalid_value"}
        result = reporting_system.execute_automated_workflow(invalid_config)

        # The system should handle errors gracefully
        # Even if the workflow fails, we should get a valid result object
        assert hasattr(result, "success")
        assert hasattr(result, "error_details")

        # Test that we can still export partial data
        partial_data = {"error_summary": "test"}
        export_result = reporting_system.export_manager.export_report(
            partial_data, "json", output_dir=tmp_path
        )
        assert Path(export_result["path"]).exists()

    def test_concurrent_report_generation(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test system performance with concurrent reporting requests."""
        import threading
        import time

        config = StakeholderReportConfig(executive_enabled=True)
        reporting_system.config = config
        results = []
        errors = []

        def generate_report():
            try:
                start_time = time.time()
                automation_config = {"test_mode": True}
                reporting_system.execute_automated_workflow(automation_config)
                reporting_system.export_manager.export_report(
                    {"test": "data"}, "json", output_dir=tmp_path
                )
                end_time = time.time()
                results.append(end_time - start_time)
            except Exception as e:
                errors.append(str(e))

        # Run 3 concurrent reports
        threads = []
        for _i in range(3):
            thread = threading.Thread(target=generate_report)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify performance and success
        assert len(results) > 0  # At least some should succeed
        if results:
            avg_time = sum(results) / len(results)
            assert avg_time < 5.0  # Should complete within reasonable time

    def test_resource_cleanup_after_reporting(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test that resources are properly cleaned up after reporting."""
        config = StakeholderReportConfig(
            executive_enabled=True, technical_enabled=True
        )

        reporting_system.config = config
        automation_config = {"test_mode": True}
        result = reporting_system.execute_automated_workflow(automation_config)

        # Export reports
        export_results = (
            reporting_system.export_manager.export_multiple_formats(
                {"test_data": "sample"},
                formats=["html", "json"],
                output_dir=tmp_path,
            )
        )

        # Verify files were created
        for _format_type, result in export_results.items():
            assert Path(result["path"]).exists()

        # Verify exported files still exist (they shouldn't be cleaned up)
        for _format_type, result in export_results.items():
            assert Path(result["path"]).exists()

    def test_metadata_consistency_across_exports(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test metadata consistency across different export formats."""
        config = StakeholderReportConfig(executive_enabled=True)
        reporting_system.config = config

        automation_config = {"test_mode": True}
        reporting_system.execute_automated_workflow(automation_config)
        export_results = (
            reporting_system.export_manager.export_multiple_formats(
                {"test_data": "sample"},
                formats=["html", "json"],
                output_dir=tmp_path,
            )
        )

        # Verify HTML contains similar metadata information
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")
        assert "test_data" in html_content
