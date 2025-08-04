"""Trend analysis for coverage monitoring.

This module handles coverage trend analysis including historical
data processing and trend calculations.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CoverageTrendAnalyzer:
    """Handles coverage trend analysis."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the trend analyzer.

        Args:
            db_path: Database path for historical data
        """
        self.db_path = db_path

    def analyze_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyze coverage trends over specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, overall_coverage, modules_above_threshold,
                       critical_gaps
                FROM coverage_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """,
                (cutoff_date,),
            )

            data = cursor.fetchall()

        if not data:
            return {"error": "No historical data available"}

        # Calculate trends
        coverages = [row[1] for row in data]
        modules_above = [row[2] for row in data]
        critical_gaps = [row[3] for row in data]

        trend_analysis = {
            "period_days": days,
            "data_points": len(data),
            "coverage_trend": {
                "start": coverages[0] if coverages else 0,
                "end": coverages[-1] if coverages else 0,
                "change": (
                    coverages[-1] - coverages[0] if len(coverages) >= 2 else 0
                ),
                "max": max(coverages) if coverages else 0,
                "min": min(coverages) if coverages else 0,
                "average": sum(coverages) / len(coverages) if coverages else 0,
            },
            "modules_trend": {
                "start": modules_above[0] if modules_above else 0,
                "end": modules_above[-1] if modules_above else 0,
                "change": (
                    modules_above[-1] - modules_above[0]
                    if len(modules_above) >= 2
                    else 0
                ),
            },
            "critical_gaps_trend": {
                "start": critical_gaps[0] if critical_gaps else 0,
                "end": critical_gaps[-1] if critical_gaps else 0,
                "change": (
                    critical_gaps[-1] - critical_gaps[0]
                    if len(critical_gaps) >= 2
                    else 0
                ),
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return trend_analysis

    def generate_trend_report(self, output_dir: Path) -> None:
        """Generate trend analysis report.

        Args:
            output_dir: Output directory for reports
        """
        trend_data = self.analyze_trends(days=30)

        if "error" in trend_data:
            logger.warning(
                f"Could not generate trend report: {trend_data['error']}"
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"trend_analysis_{timestamp}.json"
        report_path = output_dir / "reports" / report_filename

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(trend_data, f, indent=2)

        logger.info(f"Trend report generated: {report_path}")

    def get_coverage_history(self, days: int = 30) -> list[tuple[str, float]]:
        """Get coverage history for specified period.

        Args:
            days: Number of days to retrieve

        Returns:
            List of (timestamp, coverage) tuples
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, overall_coverage
                FROM coverage_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """,
                (cutoff_date,),
            )

            return cursor.fetchall()

    def calculate_average_coverage(self, days: int = 7) -> float:
        """Calculate average coverage over specified period.

        Args:
            days: Number of days to average

        Returns:
            Average coverage percentage
        """
        history = self.get_coverage_history(days)

        if not history:
            return 0.0

        coverages = [coverage for _, coverage in history]
        return sum(coverages) / len(coverages)
