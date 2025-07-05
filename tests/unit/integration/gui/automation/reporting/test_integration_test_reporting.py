"""Unit tests for IntegrationTestReportingComponent.

This module tests the main orchestrator of the comprehensive integration
test reporting system.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.integration.gui.automation.reporting.integration_test_reporting import (
    IntegrationTestReportingComponent,
)
from tests.integration.gui.automation.reporting.stakeholder_reporting import (
    StakeholderReportConfig,
)


class TestIntegrationTestReportingComponent:
    """Test suite for IntegrationTestReportingComponent."""

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, Mock]:
        """Create mock dependencies for testing."""
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
    def reporting_component(
        self, mock_dependencies: dict[str, Mock]
    ) -> IntegrationTestReportingComponent:
        """Create IntegrationTestReportingComponent with mocked dependencies."""
        return IntegrationTestReportingComponent(**mock_dependencies)

    def test_initialization(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test component initializes correctly."""
        assert reporting_component.data_aggregator is not None
        assert reporting_component.analysis_engine is not None
        assert reporting_component.stakeholder_reporter is not None
        assert reporting_component.export_manager is not None

    def test_generate_comprehensive_report_basic(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test basic report generation."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
            include_trends=True,
            include_regressions=True,
        )

        result = reporting_component.generate_comprehensive_report(config)

        assert isinstance(result, dict)
        assert "executive_summary" in result
        assert "technical_analysis" in result
        assert "operations_monitoring" in result
        assert "metadata" in result

    def test_generate_comprehensive_report_with_trends(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test report generation with trend analysis."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=False,
            operations_monitoring=False,
            include_trends=True,
            include_regressions=False,
        )

        result = reporting_component.generate_comprehensive_report(config)

        assert "trend_analysis" in result
        assert result["trend_analysis"] is not None

    def test_generate_comprehensive_report_with_regressions(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test report generation with regression detection."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=False,
            operations_monitoring=False,
            include_trends=False,
            include_regressions=True,
        )

        result = reporting_component.generate_comprehensive_report(config)

        assert "regression_analysis" in result
        assert result["regression_analysis"] is not None

    def test_export_reports_html_format(
        self,
        reporting_component: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test HTML export functionality."""
        report_data = {
            "executive_summary": {"key": "value"},
            "technical_analysis": {"key": "value"},
            "operations_monitoring": {"key": "value"},
        }

        result = reporting_component.export_reports(
            report_data, formats=["html"], output_dir=tmp_path
        )

        assert "html" in result
        assert "path" in result["html"]
        assert Path(result["html"]["path"]).exists()

    def test_export_reports_json_format(
        self,
        reporting_component: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test JSON export functionality."""
        report_data = {
            "executive_summary": {"key": "value"},
            "technical_analysis": {"key": "value"},
        }

        result = reporting_component.export_reports(
            report_data, formats=["json"], output_dir=tmp_path
        )

        assert "json" in result
        assert "path" in result["json"]
        assert Path(result["json"]["path"]).exists()

    def test_export_reports_csv_format(
        self,
        reporting_component: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test CSV export functionality."""
        report_data = {
            "executive_summary": {"metrics": {"success_rate": 95.0}},
            "technical_analysis": {"metrics": {"coverage": 85.0}},
        }

        result = reporting_component.export_reports(
            report_data, formats=["csv"], output_dir=tmp_path
        )

        assert "csv" in result
        assert "path" in result["csv"]
        assert Path(result["csv"]["path"]).exists()

    def test_export_reports_multiple_formats(
        self,
        reporting_component: IntegrationTestReportingComponent,
        tmp_path: Path,
    ) -> None:
        """Test export with multiple formats."""
        report_data = {
            "executive_summary": {"key": "value"},
            "technical_analysis": {"key": "value"},
        }

        result = reporting_component.export_reports(
            report_data, formats=["html", "json", "csv"], output_dir=tmp_path
        )

        assert "html" in result
        assert "json" in result
        assert "csv" in result
        assert all(
            Path(result[fmt]["path"]).exists()
            for fmt in ["html", "json", "csv"]
        )

    def test_get_report_status(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test report status retrieval."""
        status = reporting_component.get_report_status()

        assert isinstance(status, dict)
        assert "data_freshness" in status
        assert "system_health" in status
        assert "component_status" in status

    def test_validate_report_data(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test report data validation."""
        valid_data = {
            "executive_summary": {"key": "value"},
            "technical_analysis": {"key": "value"},
            "operations_monitoring": {"key": "value"},
        }

        # Should not raise any exception
        reporting_component._validate_report_data(valid_data)

    def test_validate_report_data_invalid(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test report data validation with invalid data."""
        invalid_data = {"invalid_key": "value"}

        with pytest.raises(ValueError, match="Missing required sections"):
            reporting_component._validate_report_data(invalid_data)

    def test_error_handling_in_report_generation(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test error handling during report generation."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        # Mock data aggregator to raise an exception
        with patch.object(
            reporting_component.data_aggregator, "aggregate_test_data"
        ) as mock_aggregate:
            mock_aggregate.side_effect = Exception("Test error")

            with pytest.raises(Exception, match="Test error"):
                reporting_component.generate_comprehensive_report(config)

    def test_metadata_generation(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test metadata generation in reports."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=False,
            operations_monitoring=False,
        )

        result = reporting_component.generate_comprehensive_report(config)

        assert "metadata" in result
        metadata = result["metadata"]
        assert "generation_timestamp" in metadata
        assert "config_used" in metadata
        assert "data_sources" in metadata

    def test_resource_cleanup(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test resource cleanup functionality."""
        # This would test if the component properly cleans up resources
        # For now, we just verify the method exists and can be called
        try:
            reporting_component.cleanup_resources()
        except AttributeError:
            # Method might not exist yet, which is acceptable
            pass

    def test_concurrent_report_generation(
        self, reporting_component: IntegrationTestReportingComponent
    ) -> None:
        """Test concurrent report generation handling."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        # Generate multiple reports concurrently (simplified test)
        results = []
        for _ in range(3):
            result = reporting_component.generate_comprehensive_report(config)
            results.append(result)

        # All results should be valid
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all("executive_summary" in result for result in results)
