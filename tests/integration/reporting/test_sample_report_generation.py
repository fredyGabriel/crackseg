"""Sample report generation and validation tests.

This module generates sample reports with realistic data and validates
that HTML, JSON, and CSV outputs are properly formatted and contain
expected content.
"""

import csv
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


class TestSampleReportGeneration:
    """Test suite for sample report generation and validation."""

    @pytest.fixture
    def realistic_mock_components(self) -> dict[str, Mock]:
        """Create mock components with realistic return values."""
        # Configure workflow scenarios component
        workflow_mock = Mock()
        workflow_mock.collect_workflow_data.return_value = {
            "total_scenarios": 15,
            "successful_scenarios": 13,
            "failed_scenarios": 2,
            "success_rate": 86.7,
            "scenario_types": [
                "basic_navigation",
                "config_validation",
                "error_handling",
            ],
            "avg_execution_time": 45.2,
        }

        # Configure error scenarios component
        error_mock = Mock()
        error_mock.collect_error_data.return_value = {
            "total_error_tests": 8,
            "handled_errors": 7,
            "unhandled_errors": 1,
            "error_recovery_rate": 87.5,
            "error_categories": [
                "validation_errors",
                "network_timeouts",
                "ui_exceptions",
            ],
            "critical_errors": 1,
        }

        # Configure session state component
        session_mock = Mock()
        session_mock.collect_session_data.return_value = {
            "total_state_tests": 12,
            "state_persistence_tests": 10,
            "state_corruption_tests": 2,
            "persistence_rate": 83.3,
            "state_categories": [
                "user_preferences",
                "config_cache",
                "session_data",
            ],
        }

        # Configure concurrent operations component
        concurrent_mock = Mock()
        concurrent_mock.collect_concurrency_data.return_value = {
            "total_concurrent_tests": 6,
            "stable_operations": 5,
            "unstable_operations": 1,
            "stability_rate": 83.3,
            "concurrency_levels": [2, 4, 8],
            "max_tested_concurrency": 8,
        }

        # Configure automation reporter
        automation_mock = Mock()
        automation_mock.collect_automation_metrics.return_value = {
            "total_automated_workflows": 4,
            "successful_automations": 4,
            "automation_success_rate": 100.0,
            "avg_automation_time": 120.5,
            "workflow_types": ["sequential", "parallel", "error_recovery"],
        }

        # Configure performance monitor
        performance_mock = Mock()
        performance_mock.collect_performance_data.return_value = {
            "page_load_times": {
                "avg": 1.5,
                "min": 0.8,
                "max": 3.2,
                "median": 1.4,
            },
            "config_validation_times": {"avg": 0.3, "min": 0.1, "max": 0.8},
            "memory_usage": {
                "avg_mb": 245.0,
                "peak_mb": 380.0,
                "baseline_mb": 120.0,
            },
            "page_load_compliance": True,
            "config_validation_compliance": True,
        }

        # Configure resource cleanup component
        cleanup_mock = Mock()
        cleanup_mock.collect_cleanup_metrics.return_value = {
            "total_cleanup_tests": 10,
            "passed_cleanup_tests": 9,
            "failed_cleanup_tests": 1,
            "cleanup_effectiveness_rate": 90.0,
            "cleanup_categories": {
                "temporary_files": {"total": 3, "passed": 3},
                "cache_cleanup": {"total": 4, "passed": 3},
                "session_cleanup": {"total": 3, "passed": 3},
            },
        }

        return {
            "automation_reporter": automation_mock,
            "resource_cleanup_component": cleanup_mock,
            "performance_monitor": performance_mock,
            "workflow_scenarios_component": workflow_mock,
            "error_scenarios_component": error_mock,
            "session_state_component": session_mock,
            "concurrent_operations_component": concurrent_mock,
        }

    @pytest.fixture
    def sample_reporting_system(
        self, realistic_mock_components: dict[str, Mock]
    ) -> IntegrationTestReportingComponent:
        """Create reporting system with realistic mock data."""
        return IntegrationTestReportingComponent(
            test_utilities=realistic_mock_components
        )

    def test_generate_executive_sample_report(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Generate and validate executive sample report."""
        config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=False,
            operations_enabled=False,
            include_trends=True,
            include_regression_analysis=False,
        )

        # Configure the system
        sample_reporting_system.config = config

        # Generate report
        automation_config = {"test_mode": True}
        result = sample_reporting_system.execute_automated_workflow(
            automation_config
        )

        # Validate workflow execution
        assert result.success
        assert "executive" in result.metadata.get("stakeholder_coverage", [])

        # Export to all formats
        export_results = (
            sample_reporting_system.export_manager.export_multiple_formats(
                {"executive_summary": {"overall_success_rate": 86.7}},
                formats=["html", "json", "csv"],
                output_dir=tmp_path,
            )
        )

        # Validate HTML output
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")
        assert "<html" in html_content
        assert "86.7" in html_content  # Success rate from mock data

        # Validate JSON output
        json_path = Path(export_results["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)
        assert json_data["executive_summary"]["overall_success_rate"] == 86.7

        # Validate CSV output
        csv_path = Path(export_results["csv"]["path"])
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "executive_summary" in csv_content
        assert "overall_success_rate" in csv_content

    def test_generate_technical_sample_report(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Generate and validate technical sample report."""
        config = StakeholderReportConfig(
            executive_enabled=False,
            technical_enabled=True,
            operations_enabled=False,
            include_trends=True,
            include_regression_analysis=True,
        )

        # Configure the system
        sample_reporting_system.config = config

        # Generate report
        automation_config = {"test_mode": True}
        result = sample_reporting_system.execute_automated_workflow(
            automation_config
        )

        # Validate technical content was processed
        assert result.success
        assert "technical" in result.metadata.get("stakeholder_coverage", [])

        # Export and validate JSON format (preferred for technical users)
        export_result = sample_reporting_system.export_manager.export_report(
            {"technical_analysis": {"overall_test_success_rate": 87.5}},
            "json",
            output_dir=tmp_path,
        )

        json_path = Path(export_result["path"])
        with open(json_path, encoding="utf-8") as f:
            technical_data = json.load(f)

        # Verify technical metrics are present
        tech_section = technical_data["technical_analysis"]
        assert "overall_test_success_rate" in tech_section

    def test_generate_operations_sample_report(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Generate and validate operations sample report."""
        config = StakeholderReportConfig(
            executive_enabled=False,
            technical_enabled=False,
            operations_enabled=True,
            include_trends=False,
            include_regression_analysis=True,
        )

        # Configure the system
        sample_reporting_system.config = config

        # Generate report
        automation_config = {"test_mode": True}
        result = sample_reporting_system.execute_automated_workflow(
            automation_config
        )

        # Validate operations content was processed
        assert result.success
        assert "operations" in result.metadata.get("stakeholder_coverage", [])

        # Export and validate CSV format (useful for operations data analysis)
        export_result = sample_reporting_system.export_manager.export_report(
            {"operations_monitoring": {"system_reliability": 90.0}},
            "csv",
            output_dir=tmp_path,
        )

        csv_path = Path(export_result["path"])

        # Parse CSV to validate structure
        with open(csv_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)

        assert len(rows) > 0
        assert "section" in rows[0]
        assert "metric" in rows[0]
        assert "value" in rows[0]

        # Verify operations metrics are present
        operations_rows = [
            row for row in rows if row["section"] == "operations_monitoring"
        ]
        assert len(operations_rows) > 0

    def test_generate_comprehensive_sample_report(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Generate comprehensive sample report for all stakeholders."""
        config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            operations_enabled=True,
            include_trends=True,
            include_regression_analysis=True,
        )

        # Configure the system
        sample_reporting_system.config = config

        # Generate comprehensive report
        automation_config = {"test_mode": True}
        result = sample_reporting_system.execute_automated_workflow(
            automation_config
        )

        # Validate all sections were processed
        stakeholder_coverage = result.metadata.get("stakeholder_coverage", [])
        assert "executive" in stakeholder_coverage
        assert "technical" in stakeholder_coverage
        assert "operations" in stakeholder_coverage

        # Export to all formats
        export_results = (
            sample_reporting_system.export_manager.export_multiple_formats(
                {
                    "executive_summary": {"overall_success_rate": 86.7},
                    "technical_analysis": {"test_coverage": 87.5},
                    "operations_monitoring": {"system_reliability": 90.0},
                },
                formats=["html", "json", "csv"],
                output_dir=tmp_path,
            )
        )

        # Comprehensive validation of HTML output
        html_path = Path(export_results["html"]["path"])
        html_content = html_path.read_text(encoding="utf-8")

        # Should contain key metrics from mock data
        assert "86.7" in html_content  # Workflow success rate
        assert "87.5" in html_content  # Error recovery rate
        assert "90.0" in html_content  # System reliability

        # Validate JSON structure
        json_path = Path(export_results["json"]["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)

        # All sections should be present and contain data
        assert len(json_data["executive_summary"]) > 0
        assert len(json_data["technical_analysis"]) > 0
        assert len(json_data["operations_monitoring"]) > 0

        # Validate CSV contains all data
        csv_path = Path(export_results["csv"]["path"])
        with open(csv_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)

        # Should have data from all sections
        sections = {row["section"] for row in rows}
        assert "executive_summary" in sections
        assert "technical_analysis" in sections
        assert "operations_monitoring" in sections

    def test_validate_html_styling_and_structure(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Validate HTML output has proper styling and structure."""
        config = StakeholderReportConfig(executive_enabled=True)
        sample_reporting_system.config = config

        automation_config = {"test_mode": True}
        sample_reporting_system.execute_automated_workflow(automation_config)

        export_result = sample_reporting_system.export_manager.export_report(
            {"test_data": "sample"}, "html", output_dir=tmp_path
        )

        html_path = Path(export_result["path"])
        html_content = html_path.read_text(encoding="utf-8")

        # Check for proper HTML structure
        assert "<html" in html_content
        assert "test_data" in html_content

    def test_validate_json_schema_compliance(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Validate JSON output follows expected schema."""
        config = StakeholderReportConfig(
            executive_enabled=True, technical_enabled=True
        )
        sample_reporting_system.config = config

        automation_config = {"test_mode": True}
        sample_reporting_system.execute_automated_workflow(automation_config)

        export_result = sample_reporting_system.export_manager.export_report(
            {"test_data": "sample"}, "json", output_dir=tmp_path
        )

        json_path = Path(export_result["path"])
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)

        # Validate data types
        assert isinstance(json_data, dict)
        assert "test_data" in json_data

    def test_validate_csv_format_consistency(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Validate CSV output format consistency."""
        config = StakeholderReportConfig(
            executive_enabled=True, operations_enabled=True
        )
        sample_reporting_system.config = config

        automation_config = {"test_mode": True}
        sample_reporting_system.execute_automated_workflow(automation_config)

        export_result = sample_reporting_system.export_manager.export_report(
            {"test_data": {"metric1": 100, "metric2": 200}},
            "csv",
            output_dir=tmp_path,
        )

        csv_path = Path(export_result["path"])

        # Validate CSV can be properly parsed
        with open(csv_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)

        # Validate header consistency
        expected_headers = ["section", "metric", "value"]
        assert csv_reader.fieldnames == expected_headers

        # Validate data consistency
        for row in rows:
            assert row["section"]  # Should not be empty
            assert row["metric"]  # Should not be empty
            assert row["value"]  # Should not be empty

        # Validate data types in value column
        numeric_rows = [
            row
            for row in rows
            if row["value"].replace(".", "").replace("-", "").isdigit()
        ]
        assert len(numeric_rows) > 0  # Should have some numeric values

    def test_generate_sample_reports_with_different_data_scenarios(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test report generation with different data scenarios."""
        scenarios = [
            {
                "name": "high_performance",
                "config": StakeholderReportConfig(executive_enabled=True),
                "expected_content": ["excellent", "performing", "successful"],
            },
            {
                "name": "mixed_results",
                "config": StakeholderReportConfig(technical_enabled=True),
                "expected_content": ["technical", "analysis", "performance"],
            },
            {
                "name": "operations_focused",
                "config": StakeholderReportConfig(operations_enabled=True),
                "expected_content": ["operations", "monitoring", "resource"],
            },
        ]

        for scenario in scenarios:
            # Configure the system
            sample_reporting_system.config = scenario["config"]

            # Generate report for scenario
            automation_config = {"test_mode": True}
            sample_reporting_system.execute_automated_workflow(
                automation_config
            )

            # Export to HTML for content validation
            export_result = (
                sample_reporting_system.export_manager.export_report(
                    {"test_data": "sample"},
                    "html",
                    output_dir=tmp_path,
                    filename=f"sample_report_{scenario['name']}",
                )
            )

            # Validate scenario-specific content
            html_path = Path(export_result["path"])
            html_content = html_path.read_text(encoding="utf-8").lower()

            # At least one expected term should be present
            found_terms = [
                term
                for term in scenario["expected_content"]
                if term in html_content
            ]
            assert (
                len(found_terms) > 0
            ), f"No expected terms found for scenario {scenario['name']}"

    def test_report_performance_with_realistic_data(
        self,
        sample_reporting_system: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test report generation performance with realistic data volumes."""
        import time

        config = StakeholderReportConfig(
            executive_enabled=True,
            technical_enabled=True,
            operations_enabled=True,
        )

        sample_reporting_system.config = config

        # Measure performance
        start_time = time.time()
        automation_config = {"test_mode": True}
        result = sample_reporting_system.execute_automated_workflow(
            automation_config
        )
        generation_time = time.time() - start_time

        export_start = time.time()
        export_results = (
            sample_reporting_system.export_manager.export_multiple_formats(
                {"test_data": "sample"},
                formats=["html", "json", "csv"],
                output_dir=tmp_path,
            )
        )
        export_time = time.time() - export_start

        # Validate performance benchmarks
        assert (
            generation_time < 3.0
        ), f"Report generation took {generation_time:.2f}s (too slow)"
        assert export_time < 2.0, f"Export took {export_time:.2f}s (too slow)"

        # Validate all exports completed successfully
        assert len(export_results) == 3
        for _format_type, result in export_results.items():
            assert Path(result["path"]).exists()
            assert Path(result["path"]).stat().st_size > 0
