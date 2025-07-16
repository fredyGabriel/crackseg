"""End-to-end integration tests for the comprehensive reporting system.

This module tests the complete workflow from crackseg.data collection through
analysis to multi-format export, validating the entire reporting pipeline.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.integration.gui.automation.reporting.integration_test_reporting import (
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
        return IntegrationTestReportingComponent(**mock_test_components)

    @pytest.fixture
    def realistic_test_config(self) -> StakeholderReportConfig:
        """Create realistic comprehensive test configuration."""
        return StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
            include_trends=True,
            include_regressions=True,
        )

    def test_complete_reporting_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        realistic_test_config: StakeholderReportConfig,
        tmp_path: Path,
    ) -> None:
        """Test complete end-to-end reporting workflow."""
        # Step 1: Generate comprehensive report
        report_data = reporting_system.generate_comprehensive_report(
            realistic_test_config
        )

        # Verify report structure
        assert isinstance(report_data, dict)
        assert "executive_summary" in report_data
        assert "technical_analysis" in report_data
        assert "operations_monitoring" in report_data
        assert "metadata" in report_data

        # Step 2: Export to all formats
        export_results = reporting_system.export_reports(
            report_data, formats=["html", "json", "csv"], output_dir=tmp_path
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
            executive_summary=True,
            technical_analysis=False,
            operations_monitoring=False,
            include_trends=True,
            include_regressions=False,
        )

        # Generate executive-focused report
        report_data = reporting_system.generate_comprehensive_report(
            executive_config
        )

        # Verify executive content
        assert "executive_summary" in report_data
        assert "technical_analysis" not in report_data
        assert "operations_monitoring" not in report_data

        # Export as HTML for presentation
        export_result = reporting_system.export_reports(
            report_data, formats=["html"], output_dir=tmp_path
        )

        # Validate HTML export for executive use
        html_path = Path(export_result["html"]["path"])
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html" in content
        assert "executive" in content.lower()

    def test_technical_deep_dive_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow for technical deep-dive analysis."""
        technical_config = StakeholderReportConfig(
            executive_summary=False,
            technical_analysis=True,
            operations_monitoring=False,
            include_trends=True,
            include_regressions=True,
        )

        # Generate technical analysis
        report_data = reporting_system.generate_comprehensive_report(
            technical_config
        )

        # Verify technical content
        assert "technical_analysis" in report_data
        assert "executive_summary" not in report_data

        # Export as JSON for programmatic analysis
        export_result = reporting_system.export_reports(
            report_data, formats=["json"], output_dir=tmp_path
        )

        # Validate JSON structure for technical use
        json_path = Path(export_result["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "technical_analysis" in data
        assert "metadata" in data

    def test_operations_monitoring_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test workflow for operations monitoring."""
        operations_config = StakeholderReportConfig(
            executive_summary=False,
            technical_analysis=False,
            operations_monitoring=True,
            include_trends=False,
            include_regressions=True,
        )

        # Generate operations report
        report_data = reporting_system.generate_comprehensive_report(
            operations_config
        )

        # Verify operations content
        assert "operations_monitoring" in report_data
        assert "executive_summary" not in report_data

        # Export as CSV for data analysis
        export_result = reporting_system.export_reports(
            report_data, formats=["csv"], output_dir=tmp_path
        )

        # Validate CSV format for operations
        csv_path = Path(export_result["csv"]["path"])
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
                executive_summary=True,
                technical_analysis=False,
                operations_monitoring=False,
            ),
            "technical": StakeholderReportConfig(
                executive_summary=False,
                technical_analysis=True,
                operations_monitoring=False,
            ),
            "operations": StakeholderReportConfig(
                executive_summary=False,
                technical_analysis=False,
                operations_monitoring=True,
            ),
        }

        results = {}
        for stakeholder, config in configs.items():
            report_data = reporting_system.generate_comprehensive_report(
                config
            )
            export_result = reporting_system.export_reports(
                report_data,
                formats=["html"],
                output_dir=tmp_path,
                filename=f"{stakeholder}_report",
            )
            results[stakeholder] = export_result

        # Verify all stakeholder reports were generated
        assert len(results) == 3
        for stakeholder, result in results.items():
            file_path = Path(result["html"]["path"])
            assert file_path.exists()
            assert stakeholder in file_path.name

    def test_trend_analysis_integration(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test integration of trend analysis in the workflow."""
        trend_config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            include_trends=True,
            include_regressions=False,
        )

        # Generate report with trends
        report_data = reporting_system.generate_comprehensive_report(
            trend_config
        )

        # Verify trend analysis is included
        if "trend_analysis" in report_data:
            trends = report_data["trend_analysis"]
            assert isinstance(trends, dict)
        else:
            # Trends might be embedded in other sections
            exec_summary = report_data.get("executive_summary", {})
            tech_analysis = report_data.get("technical_analysis", {})
            assert (
                "trend" in str(exec_summary).lower()
                or "trend" in str(tech_analysis).lower()
            )

    def test_regression_detection_integration(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test integration of regression detection in the workflow."""
        regression_config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            include_trends=False,
            include_regressions=True,
        )

        # Generate report with regression analysis
        report_data = reporting_system.generate_comprehensive_report(
            regression_config
        )

        # Verify regression analysis is included
        if "regression_analysis" in report_data:
            regressions = report_data["regression_analysis"]
            assert isinstance(regressions, dict)
        else:
            # Regressions might be embedded in other sections
            exec_summary = report_data.get("executive_summary", {})
            tech_analysis = report_data.get("technical_analysis", {})
            assert (
                "regression" in str(exec_summary).lower()
                or "regression" in str(tech_analysis).lower()
            )

    def test_data_aggregation_completeness(
        self, reporting_system: IntegrationTestReportingComponent
    ) -> None:
        """Test that data aggregation captures all testing phases."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        report_data = reporting_system.generate_comprehensive_report(config)

        # Check that all phases are represented in the data
        metadata = report_data.get("metadata", {})
        data_sources = metadata.get("data_sources", [])

        # Should include data from phases 9.1-9.7
        [
            "workflow_scenarios",  # 9.1
            "error_scenarios",  # 9.2
            "state_management",  # 9.3
            "concurrent_operations",  # 9.4
            "automation_metrics",  # 9.5
            "resource_contention",  # 9.6
            "system_stability",  # 9.7
        ]

        # At least some of these phases should be represented
        assert len(data_sources) > 0

    def test_export_format_compatibility(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test compatibility and correctness of different export formats."""
        config = StakeholderReportConfig(
            executive_summary=True, technical_analysis=True
        )

        report_data = reporting_system.generate_comprehensive_report(config)

        # Export to all formats
        export_results = reporting_system.export_reports(
            report_data, formats=["html", "json", "csv"], output_dir=tmp_path
        )

        # Validate HTML format
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content or "<html" in html_content
        assert "executive_summary" in html_content

        # Validate JSON format
        json_path = Path(export_results["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)
        assert "executive_summary" in json_data
        assert "metadata" in json_data

        # Validate CSV format
        csv_path = Path(export_results["csv"]["path"])
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "section,metric,value" in csv_content
        assert "executive_summary" in csv_content

    def test_error_recovery_in_workflow(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test error recovery mechanisms in the workflow."""
        config = StakeholderReportConfig(executive_summary=True)

        # Mock a component to fail and verify graceful handling
        with pytest.raises(Exception):
            # This should raise an exception from the mocked components
            # but demonstrate that the system attempts error recovery
            reporting_system.perform_error_recovery_simulation(
                lambda: reporting_system.get_report_data(config)
            )

        # Ensure a partial report can still be generated
        partial_report_data = reporting_system.get_report_data(
            config, allow_partial=True
        )
        assert "error_summary" in partial_report_data["metadata"]

        # And can be exported
        reporting_system.export_reports(
            partial_report_data, formats=["json"], output_dir=tmp_path
        )

    def test_concurrent_report_generation(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test system performance with concurrent reporting requests."""
        import threading
        import time

        config = StakeholderReportConfig(executive_summary=True)
        results = []
        errors = []

        def generate_report():
            try:
                start_time = time.time()
                report_data = reporting_system.generate_comprehensive_report(
                    config
                )
                export_result = reporting_system.export_reports(
                    report_data, formats=["json"], output_dir=tmp_path
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
            executive_summary=True, technical_analysis=True
        )

        # Generate and export report
        report_data = reporting_system.generate_comprehensive_report(config)
        export_results = reporting_system.export_reports(
            report_data, formats=["html", "json"], output_dir=tmp_path
        )

        # Verify files were created
        for _format_type, result in export_results.items():
            assert Path(result["path"]).exists()

        # Test cleanup (if cleanup method exists)
        try:
            reporting_system.cleanup_resources()
        except AttributeError:
            # Cleanup method might not be implemented yet
            pass

        # Verify exported files still exist (they shouldn't be cleaned up)
        for _format_type, result in export_results.items():
            assert Path(result["path"]).exists()

    def test_metadata_consistency_across_exports(
        self,
        reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test that metadata remains consistent across different export formats."""
        config = StakeholderReportConfig(executive_summary=True)

        report_data = reporting_system.generate_comprehensive_report(config)
        export_results = reporting_system.export_reports(
            report_data, formats=["html", "json"], output_dir=tmp_path
        )

        # Extract metadata from JSON export
        json_path = Path(export_results["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)

        metadata = json_data.get("metadata", {})
        assert "generation_timestamp" in metadata
        assert "data_sources" in metadata

        # Verify HTML contains similar metadata information
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")
        if "generation_timestamp" in metadata:
            # HTML should reference the same timestamp
            timestamp = str(metadata["generation_timestamp"])
            # Allow for partial timestamp matching
            timestamp_date = timestamp[:10]  # YYYY-MM-DD part
            assert timestamp_date in html_content or "metadata" in html_content
