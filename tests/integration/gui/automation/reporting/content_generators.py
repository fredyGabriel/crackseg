"""Content generators for stakeholder-specific reporting.

This module provides specialized content generation for different stakeholder
types (executive, technical, operations) with tailored formatting and focus.
"""

import json
from typing import Any


class StakeholderContentGenerator:
    """Generator for stakeholder-specific content."""

    def generate_stakeholder_content(
        self,
        stakeholder: str,
        report_data: dict[str, Any],
        analysis_results: dict[str, Any],
    ) -> str:
        """Generate content for specific stakeholder.

        Args:
            stakeholder: Stakeholder type
            report_data: Report data
            analysis_results: Analysis results

        Returns:
            HTML content for stakeholder
        """
        if stakeholder == "executive":
            return self._generate_executive_content(
                report_data, analysis_results
            )
        elif stakeholder == "technical":
            return self._generate_technical_content(
                report_data, analysis_results
            )
        elif stakeholder == "operations":
            return self._generate_operations_content(
                report_data, analysis_results
            )
        else:
            return self._generate_generic_content(
                report_data, analysis_results
            )

    def _generate_executive_content(
        self, report_data: dict[str, Any], analysis_results: dict[str, Any]
    ) -> str:
        """Generate executive-specific content."""
        key_metrics = report_data.get("key_metrics", {})
        recommendations = report_data.get("recommendations", [])
        business_impact = report_data.get("business_impact", {})

        # Extract business impact values to shorten template lines
        deploy_status = business_impact.get("deployment_readiness", "")
        deploy_text = business_impact.get("deployment_readiness", "Unknown")
        ux_status = business_impact.get("user_experience_impact", "")
        ux_text = business_impact.get("user_experience_impact", "Unknown")
        risk_status = business_impact.get("operational_risk", "")
        risk_text = business_impact.get("operational_risk", "Unknown")

        return f"""
        <div class="executive-summary">
            <div class="metrics-grid">
                <div class="metric-card success-rate">
                    <h3>Overall Success Rate</h3>
                    <div class="metric-value">
                        {key_metrics.get("overall_success_rate", 0):.1f}%
                    </div>
                </div>
                <div class="metric-card performance">
                    <h3>Performance Status</h3>
                    <div class="metric-value">
                        {key_metrics.get("performance_status", "Unknown")}
                    </div>
                </div>
                <div class="metric-card efficiency">
                    <h3>Resource Efficiency</h3>
                    <div class="metric-value">
                        {key_metrics.get("resource_efficiency", "Unknown")}
                    </div>
                </div>
            </div>

            <div class="business-impact">
                <h2>Business Impact</h2>
                <div class="impact-grid">
                    <div class="impact-item">
                        <span class="label">Deployment Readiness:</span>
                        <span class="value {deploy_status}">
                            {deploy_text}
                        </span>
                    </div>
                    <div class="impact-item">
                        <span class="label">User Experience Impact:</span>
                        <span class="value {ux_status}">
                            {ux_text}
                        </span>
                    </div>
                    <div class="impact-item">
                        <span class="label">Operational Risk:</span>
                        <span class="value {risk_status}">
                            {risk_text}
                        </span>
                    </div>
                </div>
            </div>

            <div class="recommendations">
                <h2>Key Recommendations</h2>
                <ul>
                    {self._format_list_items(recommendations)}
                </ul>
            </div>
        </div>
        """

    def _generate_technical_content(
        self, report_data: dict[str, Any], analysis_results: dict[str, Any]
    ) -> str:
        """Generate technical-specific content."""
        detailed_metrics = report_data.get("detailed_metrics", {})
        optimization_opportunities = report_data.get(
            "optimization_opportunities", []
        )

        # Extract metrics for shorter template lines
        workflow_metrics = detailed_metrics.get("workflow_coverage", {})
        perf_metrics = detailed_metrics.get("performance_bottlenecks", {})
        resource_metrics = detailed_metrics.get("resource_utilization", {})
        trend_data = analysis_results.get("trend_analysis", {})

        return f"""
        <div class="technical-analysis">
            <div class="metrics-section">
                <h2>Technical Metrics</h2>
                <div class="technical-grid">
                    <div class="tech-metric">
                        <h3>Workflow Coverage</h3>
                        <pre>{json.dumps(workflow_metrics, indent=2)}</pre>
                    </div>
                    <div class="tech-metric">
                        <h3>Performance Analysis</h3>
                        <pre>{json.dumps(perf_metrics, indent=2)}</pre>
                    </div>
                    <div class="tech-metric">
                        <h3>Resource Utilization</h3>
                        <pre>{json.dumps(resource_metrics, indent=2)}</pre>
                    </div>
                </div>
            </div>

            <div class="optimization-section">
                <h2>Optimization Opportunities</h2>
                <ul>
                    {self._format_list_items(optimization_opportunities)}
                </ul>
            </div>

            <div class="analysis-section">
                <h2>Trend Analysis</h2>
                <pre>{json.dumps(trend_data, indent=2)}</pre>
            </div>
        </div>
        """

    def _generate_operations_content(
        self, report_data: dict[str, Any], analysis_results: dict[str, Any]
    ) -> str:
        """Generate operations-specific content."""
        monitoring_metrics = report_data.get("monitoring_metrics", {})
        alerts = report_data.get("alerts_and_warnings", [])
        maintenance_recs = report_data.get("maintenance_recommendations", [])

        # Extract monitoring metrics for shorter template lines
        health_status = monitoring_metrics.get("resource_health", "").lower()
        health_text = monitoring_metrics.get("resource_health", "Unknown")
        cleanup_rate = monitoring_metrics.get("cleanup_effectiveness", 0)
        stability_text = monitoring_metrics.get(
            "concurrent_stability", "Unknown"
        )

        return f"""
        <div class="operations-dashboard">
            <div class="monitoring-section">
                <h2>System Health Monitoring</h2>
                <div class="health-grid">
                    <div class="health-metric">
                        <h3>Resource Health</h3>
                        <div class="status-indicator {health_status}">
                            {health_text}
                        </div>
                    </div>
                    <div class="health-metric">
                        <h3>Cleanup Effectiveness</h3>
                        <div class="metric-value">
                            {cleanup_rate:.1f}%
                        </div>
                    </div>
                    <div class="health-metric">
                        <h3>Concurrent Stability</h3>
                        <div class="status-indicator stable">
                            {stability_text}
                        </div>
                    </div>
                </div>
            </div>

            <div class="alerts-section">
                <h2>Alerts & Warnings</h2>
                {self._format_alerts(alerts)}
            </div>

            <div class="maintenance-section">
                <h2>Maintenance Recommendations</h2>
                <ul>
                    {self._format_list_items(maintenance_recs)}
                </ul>
            </div>
        </div>
        """

    def _generate_generic_content(
        self, report_data: dict[str, Any], analysis_results: dict[str, Any]
    ) -> str:
        """Generate generic content fallback."""
        return f"""
        <div class="generic-content">
            <h2>Report Data</h2>
            <pre>{json.dumps(report_data, indent=2)}</pre>

            <h2>Analysis Results</h2>
            <pre>{json.dumps(analysis_results, indent=2)}</pre>
        </div>
        """

    def _format_list_items(self, items: list[str]) -> str:
        """Format list items for HTML."""
        return "".join(f"<li>{item}</li>" for item in items)

    def _format_alerts(self, alerts: list[str]) -> str:
        """Format alerts for HTML display."""
        if not alerts:
            return '<div class="alert-item success">No alerts</div>'

        alert_html = ""
        for alert in alerts:
            alert_html += f'<div class="alert-item warning">{alert}</div>'
        return alert_html
