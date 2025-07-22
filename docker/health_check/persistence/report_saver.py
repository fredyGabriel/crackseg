"""Health report persistence."""

import json
import logging
from pathlib import Path
from typing import Any

from ..models import SystemHealthReport


class ReportSaver:
    """Save health reports to file."""

    def __init__(self) -> None:
        """Initialize report saver."""
        self.logger = logging.getLogger("report_saver")

    def save_report(
        self, report: SystemHealthReport, output_path: Path
    ) -> None:
        """
        Save health report to file. Args: report: Health report to save
        output_path: Output file path
        """
        # Convert to serializable format
        report_data: dict[str, Any] = {
            "timestamp": report.timestamp.isoformat(),
            "overall_status": report.overall_status.value,
            "dependencies_satisfied": report.dependencies_satisfied,
            "recommendations": report.recommendations,
            "metrics": report.metrics,
            "services": {},
        }

        for name, result in report.services.items():
            report_data["services"][name] = {
                "status": result.status.value,
                "response_time": result.response_time,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details,
                "error_message": result.error_message,
            }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.logger.info("Health report saved to %s", output_path)
