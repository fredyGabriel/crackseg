"""Shared utilities for debug artifacts modules."""

import sys
from pathlib import Path
from typing import Any

# Type definitions
type IssueList = list[str]
type DiagnosticResult = dict[str, Any]

# Add src to path for import s
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def validate_metrics_file(metrics_path: Path) -> DiagnosticResult:
    """
    Validate a single metrics file. Args: metrics_path: Path to the
    metrics file to validate Returns: Dictionary containing validation
    results
    """
    import json

    issues: IssueList = []

    try:
        if metrics_path.suffix == ".json":
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict | list):
                issues.append(
                    "JSON file does not contain valid data structure"
                )
            elif isinstance(data, dict) and not data:
                issues.append("JSON file is empty")
            elif isinstance(data, list) and not data:
                issues.append("JSON file contains empty array")

        elif metrics_path.suffix == ".jsonl":
            with open(metrics_path, encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                issues.append("JSONL file is empty")
            else:
                for i, line in enumerate(lines[:5]):  # Check first 5 lines
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError:
                        issues.append(f"Invalid JSON on line {i + 1}")
                        break

        elif metrics_path.suffix == ".csv":
            # Basic CSV validation (file should not be empty)
            file_size = metrics_path.stat().st_size
            if file_size == 0:
                issues.append("CSV file is empty")
            elif file_size < 50:  # Probably just headers
                issues.append("CSV file appears to contain only headers")

        return {
            "file": str(metrics_path),
            "status": "valid" if not issues else "invalid",
            "issues": issues,
            "size_bytes": metrics_path.stat().st_size,
        }

    except Exception as e:
        issues.append(f"Failed to validate metrics file: {e}")
        return {
            "file": str(metrics_path),
            "status": "corrupted",
            "issues": issues,
        }
