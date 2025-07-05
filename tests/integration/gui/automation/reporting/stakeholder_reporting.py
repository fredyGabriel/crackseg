"""Stakeholder-specific reporting functionality.

This module provides stakeholder-specific report generation capabilities
for executive, technical, and operations teams with tailored insights
and recommendations.
"""

from typing import Any

from .analysis.executive_analysis import ExecutiveAnalyzer
from .analysis.operations_analysis import OperationsAnalyzer
from .analysis.technical_analysis import TechnicalAnalyzer


class StakeholderReportConfig:
    """Configuration for stakeholder-specific reporting."""

    def __init__(
        self,
        executive_enabled: bool = True,
        technical_enabled: bool = True,
        operations_enabled: bool = True,
        include_trends: bool = True,
        include_regression_analysis: bool = True,
        export_formats: list[str] | None = None,
    ) -> None:
        """Initialize stakeholder report configuration.

        Args:
            executive_enabled: Generate executive summary reports
            technical_enabled: Generate technical detailed reports
            operations_enabled: Generate operations monitoring reports
            include_trends: Include trend analysis
            include_regression_analysis: Include regression detection
            export_formats: Export formats (html, json, csv)
        """
        self.executive_enabled = executive_enabled
        self.technical_enabled = technical_enabled
        self.operations_enabled = operations_enabled
        self.include_trends = include_trends
        self.include_regression_analysis = include_regression_analysis
        self.export_formats = export_formats or ["html", "json", "csv"]


class StakeholderReportGenerator:
    """Generator for stakeholder-specific reports."""

    def __init__(self, config: StakeholderReportConfig) -> None:
        """Initialize stakeholder report generator.

        Args:
            config: Stakeholder report configuration
        """
        self.config = config

        # Initialize specialized analyzers
        self.executive_analyzer = ExecutiveAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.operations_analyzer = OperationsAnalyzer()

    def generate_all_reports(
        self, aggregated_data: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Generate all enabled stakeholder reports.

        Args:
            aggregated_data: Aggregated testing data

        Returns:
            Dictionary of stakeholder reports
        """
        reports = {}

        if self.config.executive_enabled:
            reports["executive"] = self.generate_executive_report(
                aggregated_data
            )

        if self.config.technical_enabled:
            reports["technical"] = self.generate_technical_report(
                aggregated_data
            )

        if self.config.operations_enabled:
            reports["operations"] = self.generate_operations_report(
                aggregated_data
            )

        return reports

    def generate_executive_report(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate executive summary report.

        Args:
            data: Aggregated testing data

        Returns:
            Executive report data
        """
        analyzer = self.executive_analyzer

        return {
            "success": True,
            "title": "CrackSeg Integration Testing Executive Summary",
            "key_metrics": {
                "overall_success_rate": (
                    analyzer.calculate_overall_success_rate(data)
                ),
                "critical_issues": analyzer.identify_critical_issues(data),
                "performance_status": analyzer.assess_performance_status(data),
                "resource_efficiency": analyzer.assess_resource_efficiency(
                    data
                ),
            },
            "recommendations": analyzer.generate_executive_recommendations(
                data
            ),
            "trend_indicators": analyzer.generate_trend_indicators(data),
            "business_impact": analyzer.assess_business_impact(data),
        }

    def generate_technical_report(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate technical detailed report.

        Args:
            data: Aggregated testing data

        Returns:
            Technical report data
        """
        analyzer = self.technical_analyzer

        return {
            "success": True,
            "title": "CrackSeg Integration Testing Technical Analysis",
            "detailed_metrics": {
                "workflow_coverage": analyzer.analyze_workflow_coverage(data),
                "error_patterns": analyzer.analyze_error_patterns(data),
                "performance_bottlenecks": (
                    analyzer.analyze_performance_bottlenecks(data)
                ),
                "resource_utilization": analyzer.analyze_resource_utilization(
                    data
                ),
            },
            "optimization_opportunities": (
                analyzer.identify_optimization_opportunities(data)
            ),
            "architecture_insights": analyzer.generate_architecture_insights(
                data
            ),
            "technical_debt": analyzer.assess_technical_debt(data),
            "code_quality_metrics": analyzer.analyze_code_quality(data),
        }

    def generate_operations_report(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate operations monitoring report.

        Args:
            data: Aggregated testing data

        Returns:
            Operations report data
        """
        analyzer = self.operations_analyzer

        return {
            "success": True,
            "title": "CrackSeg Integration Testing Operations Report",
            "monitoring_metrics": {
                "resource_health": analyzer.assess_resource_health(data),
                "cleanup_effectiveness": analyzer.assess_cleanup_effectiveness(
                    data
                ),
                "concurrent_stability": analyzer.assess_concurrent_stability(
                    data
                ),
                "maintenance_indicators": analyzer.identify_maintenance_needs(
                    data
                ),
            },
            "alerts_and_warnings": analyzer.generate_operational_alerts(data),
            "maintenance_recommendations": (
                analyzer.generate_maintenance_recommendations(data)
            ),
            "capacity_planning": analyzer.analyze_capacity_requirements(data),
            "sla_compliance": analyzer.assess_sla_compliance(data),
        }
