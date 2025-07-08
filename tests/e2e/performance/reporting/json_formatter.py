"""JSON formatter for performance reports.

This module handles the generation of JSON format reports for programmatic
access and data interchange.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonFormatter:
    """Handles formatting and generation of JSON reports."""

    def __init__(self, storage_path: Path) -> None:
        """Initialize JSON formatter with storage configuration."""
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

    def generate_json_report(self, report_content: dict[str, Any]) -> Path:
        """Generate JSON format report for programmatic access."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        json_path = self.storage_path / f"performance_report_{timestamp}.json"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_content, f, indent=2, default=str)
            self.logger.info(f"JSON report generated: {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            raise

        return json_path

    def generate_structured_data(
        self, report_content: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate structured data format for API responses."""
        structured_data = {
            "report_id": self._generate_report_id(report_content),
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "1.0",
            "data": report_content,
        }
        return structured_data

    def _generate_report_id(self, report_content: dict[str, Any]) -> str:
        """Generate a unique report ID based on content."""
        metadata = report_content.get("metadata", {})
        commit_sha = metadata.get("commit_sha", "unknown")[:8]
        timestamp = metadata.get("generation_timestamp", "unknown")

        # Create a short hash-like ID
        if isinstance(timestamp, str) and len(timestamp) > 10:
            timestamp_part = timestamp.replace(":", "").replace("-", "")[:8]
        else:
            timestamp_part = "unknown"

        return f"perf-{commit_sha}-{timestamp_part}"
