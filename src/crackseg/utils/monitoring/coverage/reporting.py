"""Report generation for coverage monitoring.

This module handles the generation of various coverage reports
including summary reports, trend reports, and recommendations.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CoverageMetrics

logger = logging.getLogger(__name__)


class CoverageReporter:
    """Handles coverage report generation."""

    def __init__(self, output_dir: Path, target_threshold: float) -> None:
        """Initialize the reporter.

        Args:
            output_dir: Output directory for reports
            target_threshold: Target coverage threshold
        """
        self.output_dir = output_dir
        self.target_threshold = target_threshold

    def generate_summary_report(
        self, metrics: CoverageMetrics, trend_data: dict[str, Any]
    ) -> None:
        """Generate comprehensive coverage summary report.

        Args:
            metrics: Current coverage metrics
            trend_data: Trend analysis data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"coverage_summary_{timestamp}.json"
        report_path = self.output_dir / "reports" / report_filename

        report = {
            "current_metrics": asdict(metrics),
            "target_threshold": self.target_threshold,
            "status": (
                "PASS"
                if metrics.overall_coverage >= self.target_threshold
                else "FAIL"
            ),
            "gap_to_target": self.target_threshold - metrics.overall_coverage,
            "trend_analysis": trend_data,
            "recommendations": self._generate_recommendations(metrics),
            "generated_at": datetime.now().isoformat(),
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Summary report generated: {report_path}")

    def _generate_recommendations(self, metrics: CoverageMetrics) -> list[str]:
        """Generate actionable recommendations based on coverage metrics.

        Args:
            metrics: Coverage metrics to analyze

        Returns:
            List of recommendation strings
        """
        recommendations: list[str] = []

        if metrics.overall_coverage < self.target_threshold:
            gap = self.target_threshold - metrics.overall_coverage
            recommendations.append(
                f"Increase overall coverage by {gap:.1f}% to reach target"
            )

        if metrics.critical_gaps > 0:
            recommendations.append(
                f"Address {metrics.critical_gaps} modules with <50% coverage"
            )

        coverage_ratio = (
            metrics.modules_above_threshold / metrics.modules_count
            if metrics.modules_count > 0
            else 0
        )
        if coverage_ratio < 0.8:
            recommendations.append(
                "Focus on bringing more modules above threshold coverage"
            )

        if metrics.missing_statements > 1000:
            recommendations.append(
                "Prioritize high-impact modules to reduce missing statements "
                "efficiently"
            )

        return recommendations

    def generate_trend_report(self, trend_data: dict[str, Any]) -> None:
        """Generate detailed trend analysis report.

        Args:
            trend_data: Trend analysis data
        """
        report_path = (
            self.output_dir
            / "reports"
            / f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(trend_data, f, indent=2)

        logger.info(f"Trend report generated: {report_path}")

    def generate_coverage_badge(self, metrics: CoverageMetrics) -> Path:
        """Generate coverage badge for README.

        Args:
            metrics: Coverage metrics for badge generation

        Returns:
            Path to generated badge file
        """
        coverage = metrics.overall_coverage

        # Determine badge color based on coverage
        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellow"
        elif coverage >= 60:
            color = "orange"
        else:
            color = "red"

        # Generate badge URL
        badge_url = (
            f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"
        )

        # Save badge info to file
        badge_file = self.output_dir / "coverage_badge.json"
        badge_data = {
            "coverage": coverage,
            "color": color,
            "badge_url": badge_url,
            "generated_at": datetime.now().isoformat(),
        }

        badge_file.parent.mkdir(parents=True, exist_ok=True)
        with open(badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)

        logger.info(f"Coverage badge generated: {badge_file}")
        return badge_file
