#!/usr/bin/env python3
"""
Comprehensive Test Failure Analysis for CrackSeg Project. This module
provides detailed analysis of all 30 test failures captured from the
GUI test suite execution, categorizing errors systematically and
generating actionable insights for test improvement. Subtask 6.1 -
Test Failure Data Collection and Logging
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FailureCategory(Enum):
    """Categories of test failures based on root cause analysis."""

    IMPORT_ERROR = "import _error"
    MOCK_ERROR = "mock_error"
    CONFIG_ERROR = "config_error"
    ASSERTION_ERROR = "assertion_error"
    STREAMLIT_ERROR = "streamlit_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    TEST_INFRASTRUCTURE_ERROR = "test_infrastructure_error"


class ErrorSeverity(Enum):
    """Severity levels for test failures."""

    CRITICAL = "critical"  # Blocking basic functionality
    HIGH = "high"  # Major features broken
    MEDIUM = "medium"  # Specific functionality issues
    LOW = "low"  # Minor issues or flaky tests


@dataclass
class FailureAnalysisData:
    """Comprehensive analysis of all captured test failures."""

    test_name: str
    test_file: str
    failure_type: str
    error_message: str
    category: FailureCategory
    severity: ErrorSeverity
    root_cause: str
    suggested_fix: str
    affected_component: str
    priority_order: int


def analyze_all_failures() -> list[FailureAnalysisData]:
    """Analyze all 30 test failures captured from pytest execution."""
    data_file = Path(__file__).parent / "failure_data.json"
    with open(data_file, encoding="utf-8") as f:
        failures_data = json.load(f)

    failures = [
        FailureAnalysisData(
            test_name=item["test_name"],
            test_file=item["test_file"],
            failure_type=item["failure_type"],
            error_message=item["error_message"],
            category=FailureCategory[item["category"]],
            severity=ErrorSeverity[item["severity"]],
            root_cause=item["root_cause"],
            suggested_fix=item["suggested_fix"],
            affected_component=item["affected_component"],
            priority_order=item["priority_order"],
        )
        for item in failures_data
    ]
    return failures


def generate_comprehensive_reports() -> None:
    """
    Generate comprehensive JSON and CSV reports with all failure analysis.
    """

    failures = analyze_all_failures()

    # Sort by priority for systematic fixing
    failures_by_priority = sorted(failures, key=lambda x: x.priority_order)

    # Generate summary statistics
    summary: dict[str, Any] = {
        "analysis_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_failures_analyzed": len(failures),
            "analysis_scope": "Complete GUI test suite failure analysis",
            "execution_details": {
                "total_tests_run": 287,
                "total_failures": 30,
                "total_passed": 256,
                "total_skipped": 1,
                "failure_rate_percent": 10.45,
                "execution_time_seconds": 27.01,
            },
        },
        "category_breakdown": {},
        "severity_breakdown": {},
        "priority_breakdown": {},
        "affected_components": {},
        "systematic_issues": {},
    }

    # Calculate breakdowns
    for failure in failures:
        # Category breakdown
        cat = failure.category.value
        cat_count = summary["category_breakdown"].get(cat, 0)
        summary["category_breakdown"][cat] = cat_count + 1

        # Severity breakdown
        sev = failure.severity.value
        sev_count = summary["severity_breakdown"].get(sev, 0)
        summary["severity_breakdown"][sev] = sev_count + 1

        # Component breakdown
        comp = failure.affected_component
        comp_count = summary["affected_components"].get(comp, 0)
        summary["affected_components"][comp] = comp_count + 1

    # Identify systematic issues
    summary["systematic_issues"] = {
        "mock_session_state_issues": {
            "count": 2,
            "description": (
                "MockSessionState not properly implementing iterable interface"
            ),
            "affected_tests": ["TestConfigPage", "TestTrainPage"],
            "fix_priority": "HIGH - Fix once, resolves multiple tests",
        },
        "session_state_management": {
            "count": 4,
            "description": "Core session state management logic broken",
            "affected_tests": ["Multiple SessionStateManager tests"],
            "fix_priority": "HIGH - Core functionality",
        },
        "test_infrastructure_automation": {
            "count": 6,
            "description": (
                "Test utilities infrastructure missing temp_path attribute"
            ),
            "affected_tests": ["All ResourceCleanupValidationComponent tests"],
            "fix_priority": "HIGH - Systemic infrastructure issue",
        },
        "api_changes_from_refactoring": {
            "count": 8,
            "description": (
                "Module APIs changed during GUI refactoring, tests not updated"
            ),
            "affected_tests": ["Various component and utility tests"],
            "fix_priority": "MEDIUM - Update tests to match new APIs",
        },
    }

    # Create output directory
    output_dir = Path("artifacts/reports/comprehensive_failure_analysis")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate JSON report
    json_data = {
        "summary": summary,
        "failures": [
            {
                "test_name": f.test_name,
                "test_file": f.test_file,
                "failure_type": f.failure_type,
                "error_message": f.error_message,
                "category": f.category.value,
                "severity": f.severity.value,
                "root_cause": f.root_cause,
                "suggested_fix": f.suggested_fix,
                "affected_component": f.affected_component,
                "priority_order": f.priority_order,
            }
            for f in failures_by_priority
        ],
    }

    json_path = output_dir / f"comprehensive_failures_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Generate CSV report
    csv_path = output_dir / f"comprehensive_failures_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Priority",
                "Test Name",
                "Test File",
                "Failure Type",
                "Category",
                "Severity",
                "Component",
                "Root Cause",
                "Suggested Fix",
                "Error Message",
            ]
        )

        for failure in failures_by_priority:
            writer.writerow(
                [
                    failure.priority_order,
                    failure.test_name,
                    failure.test_file,
                    failure.failure_type,
                    failure.category.value,
                    failure.severity.value,
                    failure.affected_component,
                    failure.root_cause,
                    failure.suggested_fix,
                    (
                        failure.error_message[:100] + "..."
                        if len(failure.error_message) > 100
                        else failure.error_message
                    ),
                ]
            )

    print("🚀 Comprehensive Test Failure Analysis Complete")
    print("=" * 60)
    print(f"📊 Total Failures Analyzed: {len(failures)}")
    print(
        f"📈 Failure Rate: "
        f"{summary['analysis_metadata']['execution_details']['failure_rate_percent']}%"
    )

    print("\n🏷️ CATEGORY BREAKDOWN:")
    for category, count in summary["category_breakdown"].items():
        print(f"  {category}: {count}")

    print("\n⚠️ SEVERITY BREAKDOWN:")
    for severity, count in summary["severity_breakdown"].items():
        print(f"  {severity}: {count}")

    print("\n🔧 TOP SYSTEMATIC ISSUES:")
    for issue, details in summary["systematic_issues"].items():
        print(
            f"  {issue}: {details['count']} failures - "
            f"{details['fix_priority']}"
        )

    print("\n✅ Reports generated:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 CSV: {csv_path}")


if __name__ == "__main__":
    generate_comprehensive_reports()
