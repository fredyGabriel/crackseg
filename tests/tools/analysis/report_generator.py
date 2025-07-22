from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .test_failure_analysis import AnalysisSummary, TestFailure


class ReportGenerator:
    """Generates reports from test failure analysis results."""

    def __init__(self, failures: list[TestFailure], summary: AnalysisSummary):
        self.failures = failures
        self.summary = summary

    def to_json(self, output_path: str | Path):
        """Exports the analysis to a JSON file."""
        output_path = Path(output_path)
        export_data = {
            "analysis_metadata": {
                "generated_at": self.summary.analysis_timestamp,
                "analyzer_version": "1.0.0",
                "analysis_scope": "GUI test suite",
            },
            "summary": asdict(self.summary),
            "failures": [asdict(f) for f in self.failures],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"📊 JSON report exported to: {output_path}")

    def to_csv(self, output_path: str | Path):
        """Exports the analysis to a CSV file."""
        output_path = Path(output_path)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Name",
                    "Test File",
                    "Failure Type",
                    "Category",
                    "Severity",
                    "Error Message",
                    "Affected Modules",
                    "Root Causes",
                    "Suggested Fixes",
                    "Timestamp",
                ]
            )
            for failure in self.failures:
                writer.writerow(
                    [
                        failure.test_name,
                        failure.test_file,
                        failure.failure_type,
                        failure.category.value,
                        failure.severity.value,
                        failure.error_message,
                        "; ".join(failure.affected_modules),
                        "; ".join(failure.potential_root_causes),
                        "; ".join(failure.suggested_fixes),
                        failure.timestamp,
                    ]
                )
        print(f"📈 CSV report exported to: {output_path}")
