#!/usr/bin/env python3
"""
Test Failure Analysis System for CrackSeg Project.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .pytest_executor import PytestExecutor
from .pytest_output_parser import PytestOutputParser
from .report_generator import ReportGenerator


class FailureCategory(Enum):
    """Categories of test failures based on root cause analysis."""

    IMPORT_ERROR = "import _error"
    MOCK_ERROR = "mock_error"
    CONFIG_ERROR = "config_error"
    ASSERTION_ERROR = "assertion_error"
    STREAMLIT_ERROR = "streamlit_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    TIMEOUT_ERROR = "timeout_error"
    INFRASTRUCTURE_ERROR = "test_infrastructure_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for test failures."""

    CRITICAL = "critical"  # Blocking basic functionality
    HIGH = "high"  # Major features broken
    MEDIUM = "medium"  # Specific functionality issues
    LOW = "low"  # Minor issues or flaky tests


@dataclass
class TestFailure:
    """Structured representation of a single test failure."""

    test_name: str
    test_file: str
    failure_type: str
    error_message: str
    stack_trace: str
    category: FailureCategory
    severity: ErrorSeverity
    affected_modules: list[str] = field(default_factory=list)
    potential_root_causes: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnalysisSummary:
    """Summary statistics of the test failure analysis."""

    total_tests_run: int
    total_failures: int
    total_passed: int
    total_skipped: int
    failure_rate_percent: float
    categories_breakdown: dict[str, int] = field(default_factory=dict)
    severity_breakdown: dict[str, int] = field(default_factory=dict)
    most_affected_modules: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class TestFailureAnalyzer:
    """Orchestrator for the test failure analysis process."""

    def __init__(self):
        self.failures: list[TestFailure] = []
        self.summary: AnalysisSummary | None = None

    def execute_tests_and_analyze(
        self, test_paths: list[str]
    ) -> AnalysisSummary:
        """Executes the full test analysis pipeline."""
        print("🔬 Executing comprehensive test suite with detailed logging...")

        executor = PytestExecutor(test_paths)
        output = executor.run()

        parser = PytestOutputParser()
        self.failures = parser.parse(output)

        self.summary = self._generate_summary()

        return self.summary

    def _generate_summary(self) -> AnalysisSummary:
        """Generate analysis summary with statistics."""
        total_failures = len(self.failures)

        # Count categories
        categories_breakdown: dict[str, int] = {}
        severity_breakdown: dict[str, int] = {}
        all_modules: set[str] = set()

        for failure in self.failures:
            # Category counts
            cat_name = failure.category.value
            categories_breakdown[cat_name] = (
                categories_breakdown.get(cat_name, 0) + 1
            )
            # Severity counts
            sev_name = failure.severity.value
            severity_breakdown[sev_name] = (
                severity_breakdown.get(sev_name, 0) + 1
            )
            # Collect all modules
            all_modules.update(failure.affected_modules)

        # Mock summary values (would be extracted from pytest output)
        total_tests_run = (
            287  # Based on output: 30 failed, 256 passed, 1 skipped
        )
        total_passed = 256
        total_skipped = 1
        failure_rate = (
            (total_failures / total_tests_run) * 100
            if total_tests_run > 0
            else 0
        )

        return AnalysisSummary(
            total_tests_run=total_tests_run,
            total_failures=total_failures,
            total_passed=total_passed,
            total_skipped=total_skipped,
            failure_rate_percent=round(failure_rate, 2),
            categories_breakdown=categories_breakdown,
            severity_breakdown=severity_breakdown,
            most_affected_modules=sorted(all_modules)[:10],  # Top 10
            execution_time_seconds=27.01,  # From pytest output
        )

    def export_reports(self, output_dir: str | Path, timestamp: str):
        """Exports all analysis reports."""
        if not self.summary:
            raise RuntimeError(
                "Analysis must be run before exporting reports."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        generator = ReportGenerator(self.failures, self.summary)

        json_path = output_dir / f"test_failures_{timestamp}.json"
        csv_path = output_dir / f"test_failures_{timestamp}.csv"

        generator.to_json(json_path)
        generator.to_csv(csv_path)


def main() -> None:
    """Main execution function for test failure analysis."""
    print("🚀 CrackSeg Test Failure Analysis System")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TestFailureAnalyzer()

    try:
        # For demonstration, we'll use the already captured failure data
        # In practice, this would call
        # analyzer.execute_tests_and_analyze(test_paths)

        # Process known failures from previous execution
        print("📋 Processing test failure data from recent execution...")

        # Simulate the failures we captured
        _create_demo_failures_for_analyzer(analyzer)
        analyzer.summary = analyzer._generate_summary()

        # Generate reports
        output_dir = Path("test_failure_reports")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        analyzer.export_reports(output_dir, timestamp)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Tests Run: {analyzer.summary.total_tests_run}")
        print(f"Total Failures: {analyzer.summary.total_failures}")
        print(f"Failure Rate: {analyzer.summary.failure_rate_percent}%")
        print(f"Execution Time: {analyzer.summary.execution_time_seconds}s")

        print("\n🏷️ FAILURE CATEGORIES:")
        for category, count in analyzer.summary.categories_breakdown.items():
            print(f"  {category}: {count}")

        print("\n⚠️ SEVERITY BREAKDOWN:")
        for severity, count in analyzer.summary.severity_breakdown.items():
            print(f"  {severity}: {count}")

        print(f"\n✅ Analysis complete! Reports saved to {output_dir}/")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def _create_demo_failures_for_analyzer(analyzer: TestFailureAnalyzer) -> None:
    """Create demo failures based on the captured test execution."""
    demo_failures_data: list[dict[str, Any]] = [
        {
            "test_name": (
                "TestAdvancedConfigPage::test_page_advanced_config_basic_mock"
            ),
            "test_file": ("tests/unit/gui/pages/test_advanced_config_page.py"),
            "failure_type": "AttributeError",
            "error_message": (
                "module 'scripts.gui.pages.advanced_config_page' "
                "has no attribute 'render_advanced_config_page'"
            ),
            "category": FailureCategory.ATTRIBUTE_ERROR,
            "severity": ErrorSeverity.HIGH,
        },
        {
            "test_name": "TestConfigPage::test_page_config_basic_mock",
            "test_file": "tests/unit/gui/pages/test_config_page.py",
            "failure_type": "TypeError",
            "error_message": (
                "argument of type 'MockSessionState' is not iterable"
            ),
            "category": FailureCategory.STREAMLIT_ERROR,
            "severity": ErrorSeverity.HIGH,
        },
        {
            "test_name": "TestErrorCategorizer::test_categorize_value_error",
            "test_file": "tests/unit/gui/test_error_console.py",
            "failure_type": "AssertionError",
            "error_message": (
                "assert <ErrorSeverity.CRITICAL: "
                "'critical'> == <ErrorSeverity.WARNING: 'warning'>"
            ),
            "category": FailureCategory.ASSERTION_ERROR,
            "severity": ErrorSeverity.MEDIUM,
        },
    ]
    for failure_data in demo_failures_data:
        failure = TestFailure(
            test_name=str(failure_data["test_name"]),
            test_file=str(failure_data["test_file"]),
            failure_type=str(failure_data["failure_type"]),
            error_message=str(failure_data["error_message"]),
            stack_trace=f"Mock stack trace for {failure_data['test_name']}",
            category=cast(FailureCategory, failure_data["category"]),
            severity=cast(ErrorSeverity, failure_data["severity"]),
            affected_modules=[],
            potential_root_causes=[],
            suggested_fixes=[],
        )
        analyzer.failures.append(failure)


if __name__ == "__main__":
    main()
