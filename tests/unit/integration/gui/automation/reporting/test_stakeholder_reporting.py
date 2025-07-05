"""Unit tests for StakeholderReporting component.

This module tests the stakeholder-specific report generation system
that creates tailored reports for executive, technical, and operations teams.
"""

from typing import Any
from unittest.mock import patch

import pytest

from tests.integration.gui.automation.reporting.stakeholder_reporting import (
    StakeholderReportConfig,
    StakeholderReportGenerator,
)


class TestStakeholderReportGenerator:
    """Test suite for StakeholderReportGenerator functionality."""

    @pytest.fixture
    def report_generator(self) -> StakeholderReportGenerator:
        """Create StakeholderReportGenerator instance for testing."""
        return StakeholderReportGenerator()

    @pytest.fixture
    def sample_aggregated_data(self) -> dict[str, Any]:
        """Provide sample aggregated data for testing."""
        return {
            "workflow_scenarios": {
                "total_scenarios": 15,
                "successful_scenarios": 13,
                "failed_scenarios": 2,
                "success_rate": 86.7,
                "scenario_types": ["basic", "advanced", "edge_case"],
            },
            "error_scenarios": {
                "total_error_tests": 8,
                "handled_errors": 7,
                "unhandled_errors": 1,
                "error_recovery_rate": 87.5,
                "error_categories": ["validation", "network", "timeout"],
            },
            "session_state": {
                "total_state_tests": 12,
                "state_persistence_tests": 10,
                "state_corruption_tests": 2,
                "persistence_rate": 83.3,
                "state_categories": ["config", "user_data", "cache"],
            },
            "concurrent_operations": {
                "total_concurrent_tests": 6,
                "stable_operations": 5,
                "unstable_operations": 1,
                "stability_rate": 83.3,
                "concurrency_levels": [2, 4, 8],
            },
            "automation_metrics": {
                "total_automated_workflows": 4,
                "successful_automations": 4,
                "automation_success_rate": 100.0,
                "avg_automation_time": 120.5,
                "workflow_types": ["sequential", "parallel", "error_recovery"],
            },
            "performance_metrics": {
                "page_load_times": {"avg": 1.5, "min": 0.8, "max": 3.2},
                "config_validation_times": {
                    "avg": 0.3,
                    "min": 0.1,
                    "max": 0.8,
                },
                "memory_usage": {"avg_mb": 245.0, "peak_mb": 380.0},
                "page_load_compliance": True,
                "config_validation_compliance": True,
            },
            "resource_cleanup": {
                "total_cleanup_tests": 10,
                "passed_cleanup_tests": 9,
                "failed_cleanup_tests": 1,
                "cleanup_effectiveness_rate": 90.0,
                "cleanup_categories": {
                    "temporary_files": {"total": 3, "passed": 3}
                },
            },
        }

    @pytest.fixture
    def executive_config(self) -> StakeholderReportConfig:
        """Create executive-focused report configuration."""
        return StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=False,
            operations_monitoring=False,
            include_trends=True,
            include_regressions=False,
        )

    @pytest.fixture
    def technical_config(self) -> StakeholderReportConfig:
        """Create technical-focused report configuration."""
        return StakeholderReportConfig(
            executive_summary=False,
            technical_analysis=True,
            operations_monitoring=False,
            include_trends=True,
            include_regressions=True,
        )

    @pytest.fixture
    def operations_config(self) -> StakeholderReportConfig:
        """Create operations-focused report configuration."""
        return StakeholderReportConfig(
            executive_summary=False,
            technical_analysis=False,
            operations_monitoring=True,
            include_trends=False,
            include_regressions=True,
        )

    def test_initialization(
        self, report_generator: StakeholderReportGenerator
    ) -> None:
        """Test StakeholderReportGenerator initializes correctly."""
        assert report_generator.executive_analyzer is not None
        assert report_generator.technical_analyzer is not None
        assert report_generator.operations_analyzer is not None

    def test_generate_executive_report(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
        executive_config: StakeholderReportConfig,
    ) -> None:
        """Test executive report generation."""
        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, executive_config
        )

        assert "executive_summary" in result
        executive_report = result["executive_summary"]

        # Verify executive-specific content
        assert "overall_health_status" in executive_report
        assert "key_achievements" in executive_report
        assert "critical_issues" in executive_report
        assert "business_impact" in executive_report
        assert "recommendations" in executive_report

        # Verify executive metrics are high-level
        assert "overall_success_rate" in executive_report
        assert "deployment_readiness" in executive_report

    def test_generate_technical_report(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
        technical_config: StakeholderReportConfig,
    ) -> None:
        """Test technical report generation."""
        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, technical_config
        )

        assert "technical_analysis" in result
        technical_report = result["technical_analysis"]

        # Verify technical-specific content
        assert "test_coverage_analysis" in technical_report
        assert "performance_deep_dive" in technical_report
        assert "architecture_health" in technical_report
        assert "code_quality_metrics" in technical_report
        assert "technical_debt_assessment" in technical_report

        # Verify detailed metrics
        assert "detailed_test_results" in technical_report
        assert "performance_breakdown" in technical_report

    def test_generate_operations_report(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
        operations_config: StakeholderReportConfig,
    ) -> None:
        """Test operations report generation."""
        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, operations_config
        )

        assert "operations_monitoring" in result
        operations_report = result["operations_monitoring"]

        # Verify operations-specific content
        assert "system_reliability" in operations_report
        assert "resource_utilization" in operations_report
        assert "deployment_metrics" in operations_report
        assert "monitoring_insights" in operations_report
        assert "operational_recommendations" in operations_report

        # Verify operational metrics
        assert "uptime_analysis" in operations_report
        assert "resource_efficiency" in operations_report

    def test_comprehensive_report_generation(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test generation of comprehensive report for all stakeholders."""
        comprehensive_config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
            include_trends=True,
            include_regressions=True,
        )

        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, comprehensive_config
        )

        # All sections should be present
        assert "executive_summary" in result
        assert "technical_analysis" in result
        assert "operations_monitoring" in result

        # Verify each section has appropriate content
        assert len(result["executive_summary"]) > 0
        assert len(result["technical_analysis"]) > 0
        assert len(result["operations_monitoring"]) > 0

    def test_report_customization_by_role(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test that reports are properly customized by stakeholder role."""
        executive_config = StakeholderReportConfig(executive_summary=True)
        technical_config = StakeholderReportConfig(technical_analysis=True)

        executive_result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, executive_config
        )
        technical_result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, technical_config
        )

        # Executive report should focus on high-level metrics
        exec_summary = executive_result["executive_summary"]
        assert "business_impact" in exec_summary
        assert "deployment_readiness" in exec_summary

        # Technical report should include detailed analysis
        tech_analysis = technical_result["technical_analysis"]
        assert "detailed_test_results" in tech_analysis
        assert "performance_breakdown" in tech_analysis

        # Verify different focus areas
        assert "key_achievements" in exec_summary
        assert "code_quality_metrics" in tech_analysis

    def test_data_transformation_for_stakeholders(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test data transformation for different stakeholder needs."""
        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, config
        )

        # Executive data should be summarized
        exec_data = result["executive_summary"]
        assert "overall_success_rate" in exec_data
        assert isinstance(exec_data["overall_success_rate"], int | float)

        # Technical data should be detailed
        tech_data = result["technical_analysis"]
        assert "detailed_test_results" in tech_data
        assert isinstance(tech_data["detailed_test_results"], dict)

        # Operations data should focus on reliability
        ops_data = result["operations_monitoring"]
        assert "system_reliability" in ops_data
        assert isinstance(ops_data["system_reliability"], dict)

    def test_trend_analysis_integration(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test integration with trend analysis."""
        config = StakeholderReportConfig(
            executive_summary=True, include_trends=True
        )

        with patch.object(report_generator, "_analyze_trends") as mock_trends:
            mock_trends.return_value = {
                "performance_trends": {"direction": "improving"},
                "quality_trends": {"direction": "stable"},
            }

            result = report_generator.generate_stakeholder_reports(
                sample_aggregated_data, config
            )

            # Trends should be included in executive summary
            exec_summary = result["executive_summary"]
            assert "trend_analysis" in exec_summary or "trends" in str(
                exec_summary
            )

    def test_regression_analysis_integration(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test integration with regression analysis."""
        config = StakeholderReportConfig(
            technical_analysis=True, include_regressions=True
        )

        with patch.object(
            report_generator, "_analyze_regressions"
        ) as mock_regressions:
            mock_regressions.return_value = {
                "performance_regressions": [],
                "quality_regressions": ["Minor stability issue"],
            }

            result = report_generator.generate_stakeholder_reports(
                sample_aggregated_data, config
            )

            # Regressions should be included in technical analysis
            tech_analysis = result["technical_analysis"]
            assert (
                "regression_analysis" in tech_analysis
                or "regressions" in str(tech_analysis)
            )

    def test_empty_data_handling(
        self, report_generator: StakeholderReportGenerator
    ) -> None:
        """Test handling of empty aggregated data."""
        empty_data = {}
        config = StakeholderReportConfig(executive_summary=True)

        result = report_generator.generate_stakeholder_reports(
            empty_data, config
        )

        assert "executive_summary" in result
        exec_summary = result["executive_summary"]

        # Should provide default values for missing data
        assert "overall_health_status" in exec_summary
        assert "data_availability" in exec_summary

    def test_partial_data_handling(
        self, report_generator: StakeholderReportGenerator
    ) -> None:
        """Test handling of partial aggregated data."""
        partial_data = {
            "workflow_scenarios": {"success_rate": 85.0},
            # Missing other sections
        }
        config = StakeholderReportConfig(technical_analysis=True)

        result = report_generator.generate_stakeholder_reports(
            partial_data, config
        )

        assert "technical_analysis" in result
        tech_analysis = result["technical_analysis"]

        # Should handle missing sections gracefully
        assert "data_completeness" in tech_analysis
        assert "available_metrics" in tech_analysis

    def test_configuration_validation(
        self, report_generator: StakeholderReportGenerator
    ) -> None:
        """Test report configuration validation."""
        # Test invalid configuration (no sections enabled)
        invalid_config = StakeholderReportConfig(
            executive_summary=False,
            technical_analysis=False,
            operations_monitoring=False,
        )

        with pytest.raises(
            ValueError, match="At least one report section must be enabled"
        ):
            report_generator.generate_stakeholder_reports({}, invalid_config)

    def test_report_consistency(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test consistency of reports across multiple generations."""
        config = StakeholderReportConfig(executive_summary=True)

        result1 = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, config
        )
        result2 = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, config
        )

        # Results should be consistent for same input
        exec1 = result1["executive_summary"]
        exec2 = result2["executive_summary"]

        assert exec1["overall_success_rate"] == exec2["overall_success_rate"]
        assert exec1.keys() == exec2.keys()

    def test_performance_with_large_data(
        self, report_generator: StakeholderReportGenerator
    ) -> None:
        """Test performance with large aggregated data sets."""
        import time

        large_data = {
            "workflow_scenarios": {"success_rate": 85.0},
            **{
                f"large_section_{i}": {f"metric_{j}": j for j in range(100)}
                for i in range(10)
            },
        }

        config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        start_time = time.time()
        result = report_generator.generate_stakeholder_reports(
            large_data, config
        )
        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 2.0
        assert len(result) == 3  # All sections generated

    def test_error_handling_during_generation(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test error handling during report generation."""
        config = StakeholderReportConfig(executive_summary=True)

        # Mock analyzer to raise an exception
        with patch.object(
            report_generator.executive_analyzer, "analyze"
        ) as mock_analyzer:
            mock_analyzer.side_effect = Exception("Analysis failed")

            with pytest.raises(Exception, match="Analysis failed"):
                report_generator.generate_stakeholder_reports(
                    sample_aggregated_data, config
                )

    def test_metadata_inclusion(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test inclusion of metadata in generated reports."""
        config = StakeholderReportConfig(executive_summary=True)

        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, config
        )

        exec_summary = result["executive_summary"]
        assert "generation_metadata" in exec_summary
        metadata = exec_summary["generation_metadata"]

        assert "timestamp" in metadata
        assert "config_used" in metadata
        assert "data_sources" in metadata

    def test_recommendation_generation(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test recommendation generation for different stakeholders."""
        comprehensive_config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, comprehensive_config
        )

        # Each section should have appropriate recommendations
        assert "recommendations" in result["executive_summary"]
        assert "technical_recommendations" in result["technical_analysis"]
        assert "operational_recommendations" in result["operations_monitoring"]

        # Recommendations should be lists of strings
        exec_recs = result["executive_summary"]["recommendations"]
        assert isinstance(exec_recs, list)
        assert all(isinstance(rec, str) for rec in exec_recs)

    def test_cross_section_data_consistency(
        self,
        report_generator: StakeholderReportGenerator,
        sample_aggregated_data: dict[str, Any],
    ) -> None:
        """Test data consistency across different report sections."""
        comprehensive_config = StakeholderReportConfig(
            executive_summary=True,
            technical_analysis=True,
            operations_monitoring=True,
        )

        result = report_generator.generate_stakeholder_reports(
            sample_aggregated_data, comprehensive_config
        )

        # Overall success rate should be consistent across sections
        exec_rate = result["executive_summary"]["overall_success_rate"]
        tech_rate = result["technical_analysis"]["overall_test_success_rate"]

        # Should be same or very close (allowing for rounding)
        assert abs(exec_rate - tech_rate) < 1.0
