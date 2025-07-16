#!/usr/bin/env python3
"""Test Failure Categorization System for Subtask 6.2.

Systematically categorize the 34 failing tests into distinct failure types:
- Import/Module Errors
- Mock/Fixture Issues
- Configuration Problems
- Assertion Failures
- Streamlit-specific Issues

With detailed classification and root cause analysis for each category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MainFailureCategory(Enum):
    """Main categories for test failure classification per subtask 6.2."""

    IMPORT_MODULE_ERROR = "import_module_error"
    MOCK_FIXTURE_ISSUE = "mock_fixture_issue"
    CONFIGURATION_PROBLEM = "configuration_problem"
    ASSERTION_FAILURE = "assertion_failure"
    STREAMLIT_SPECIFIC_ISSUE = "streamlit_specific_issue"


@dataclass
class CategorizedFailure:
    """Categorized test failure with detailed analysis."""

    test_name: str
    test_file: str
    original_category: str
    new_main_category: MainFailureCategory
    failure_type: str
    error_message: str
    severity: str
    root_cause: str
    suggested_fix: str
    affected_component: str
    priority_order: int
    categorization_rationale: str = ""


@dataclass
class CategoryAnalysis:
    """Analysis of a specific failure category."""

    category: MainFailureCategory
    total_failures: int
    severity_breakdown: dict[str, int]
    common_patterns: list[str]
    root_causes: list[str]
    systemic_issues: list[str]
    prioritized_fixes: list[str]
    affected_components: list[str]


class TestFailureCategorizer:
    """
    Categorizes test failures into main categories with detailed analysis.
    """

    def __init__(self, json_file_path: str) -> None:
        """Initialize categorizer with failure data."""
        self.json_file_path = Path(json_file_path)
        self.failures: list[CategorizedFailure] = []
        self.category_analyses: dict[
            MainFailureCategory, CategoryAnalysis
        ] = {}

    def load_and_categorize(self) -> None:
        """Load failure data and categorize into main categories."""
        with open(self.json_file_path, encoding="utf-8") as f:
            data = json.load(f)

        for failure_data in data["failures"]:
            categorized_failure = self._categorize_failure(failure_data)
            self.failures.append(categorized_failure)

        self._analyze_categories()

    def _categorize_failure(
        self, failure_data: dict[str, Any]
    ) -> CategorizedFailure:
        """Categorize individual failure based on patterns and content."""
        original_category = failure_data["category"]
        error_message = failure_data["error_message"]
        failure_type = failure_data["failure_type"]
        test_file = failure_data["test_file"]

        # Main categorization logic
        main_category, rationale = self._determine_main_category(
            original_category, error_message, failure_type, test_file
        )

        return CategorizedFailure(
            test_name=failure_data["test_name"],
            test_file=test_file,
            original_category=original_category,
            new_main_category=main_category,
            failure_type=failure_type,
            error_message=error_message,
            severity=failure_data["severity"],
            root_cause=failure_data["root_cause"],
            suggested_fix=failure_data["suggested_fix"],
            affected_component=failure_data["affected_component"],
            priority_order=failure_data["priority_order"],
            categorization_rationale=rationale,
        )

    def _determine_main_category(
        self,
        original_category: str,
        error_message: str,
        failure_type: str,
        test_file: str,
    ) -> tuple[MainFailureCategory, str]:
        """Determine main category based on error patterns."""

        # Import/Module Errors
        if (
            original_category == "attribute_error"
            and "has no attribute" in error_message
            and "module" in error_message
        ):
            return MainFailureCategory.IMPORT_MODULE_ERROR, (
                "Module attribute error indicates missing imports or "
                "API changes"
            )

        # Mock/Fixture Issues
        if (
            original_category == "mock_error"
            or "Mock" in error_message
            or "Expected" in error_message
            and "to have been called" in error_message
        ):
            return MainFailureCategory.MOCK_FIXTURE_ISSUE, (
                "Mock expectation failures or fixture setup issues"
            )

        # Configuration Problems
        if (
            original_category == "test_infrastructure_error"
            or "temp_path" in error_message
            or "configuration" in error_message.lower()
        ):
            return MainFailureCategory.CONFIGURATION_PROBLEM, (
                "Test infrastructure or configuration setup problems"
            )

        # Streamlit-specific Issues
        if (
            original_category == "streamlit_error"
            or "MockSessionState" in error_message
            or "argument of type 'MockSessionState' is not iterable"
            in error_message
        ):
            return MainFailureCategory.STREAMLIT_SPECIFIC_ISSUE, (
                "Streamlit UI component testing or session state issues"
            )

        # Assertion Failures (default for remaining)
        if original_category == "assertion_error" or failure_type in [
            "AssertionError",
            "ValueError",
            "TypeError",
        ]:
            return MainFailureCategory.ASSERTION_FAILURE, (
                "Logic errors or expected vs actual value mismatches"
            )

        # Fallback to assertion failures
        return MainFailureCategory.ASSERTION_FAILURE, (
            "Unclassified error defaulting to assertion failure category"
        )

    def _analyze_categories(self) -> None:
        """Analyze each category for patterns and insights."""
        for category in MainFailureCategory:
            category_failures = [
                f for f in self.failures if f.new_main_category == category
            ]

            if not category_failures:
                continue

            # Severity breakdown
            severity_breakdown: dict[str, int] = {}
            for failure in category_failures:
                severity = failure.severity
                current_count = severity_breakdown.get(severity, 0)
                severity_breakdown[severity] = current_count + 1

            # Extract patterns and insights
            common_patterns = self._extract_common_patterns(category_failures)
            root_causes = list({f.root_cause for f in category_failures})
            systemic_issues = self._identify_systemic_issues(category_failures)
            prioritized_fixes = self._prioritize_fixes(category_failures)
            affected_components = list(
                {f.affected_component for f in category_failures}
            )

            self.category_analyses[category] = CategoryAnalysis(
                category=category,
                total_failures=len(category_failures),
                severity_breakdown=severity_breakdown,
                common_patterns=common_patterns,
                root_causes=root_causes,
                systemic_issues=systemic_issues,
                prioritized_fixes=prioritized_fixes,
                affected_components=affected_components,
            )

    def _extract_common_patterns(
        self, failures: list[CategorizedFailure]
    ) -> list[str]:
        """Extract common patterns from failures in a category."""
        patterns: list[str] = []
        error_messages = [f.error_message for f in failures]

        # Common error message patterns
        if any("has no attribute" in msg for msg in error_messages):
            patterns.append("Missing module attributes after API changes")

        if any("MockSessionState" in msg for msg in error_messages):
            patterns.append("MockSessionState compatibility issues")

        if any("temp_path" in msg for msg in error_messages):
            patterns.append("Test infrastructure fixture problems")

        if any("assert" in msg for msg in error_messages):
            patterns.append("Assertion logic mismatches")

        return patterns

    def _identify_systemic_issues(
        self, failures: list[CategorizedFailure]
    ) -> list[str]:
        """Identify systemic issues affecting multiple tests."""
        systemic_issues: list[str] = []

        # Group by similar error messages
        similar_errors: dict[str, list[CategorizedFailure]] = {}
        for failure in failures:
            key = failure.error_message[:50]  # First 50 chars as key
            if key not in similar_errors:
                similar_errors[key] = []
            similar_errors[key].append(failure)

        # Identify systemic problems
        for error_key, failure_list in similar_errors.items():
            if len(failure_list) > 1:
                systemic_issues.append(
                    f"Similar error affecting {len(failure_list)} tests: "
                    f"{error_key}..."
                )

        return systemic_issues

    def _prioritize_fixes(
        self, failures: list[CategorizedFailure]
    ) -> list[str]:
        """Prioritize fixes based on severity and impact."""
        # Sort by priority order (lower is higher priority)
        sorted_failures = sorted(failures, key=lambda f: f.priority_order)

        prioritized_fixes: list[str] = []
        for failure in sorted_failures[:3]:  # Top 3 priorities
            prioritized_fixes.append(
                f"Priority {failure.priority_order}: {failure.suggested_fix}"
            )

        return prioritized_fixes

    def generate_categorization_report(self) -> dict[str, Any]:
        """Generate comprehensive categorization report."""
        report: dict[str, Any] = {
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_failures_categorized": len(self.failures),
                "categorization_strategy": "hybrid_auto_manual_review",
                "categories_used": [cat.value for cat in MainFailureCategory],
            },
            "category_summary": {},
            "detailed_analysis": {},
            "cross_category_insights": (
                self._generate_cross_category_insights()
            ),
        }

        # Category summaries
        for category, analysis in self.category_analyses.items():
            report["category_summary"][category.value] = {
                "total_failures": analysis.total_failures,
                "severity_breakdown": analysis.severity_breakdown,
                "percentage_of_total": round(
                    (analysis.total_failures / len(self.failures)) * 100, 1
                ),
            }

        # Detailed analysis
        for category, analysis in self.category_analyses.items():
            report["detailed_analysis"][category.value] = {
                "common_patterns": analysis.common_patterns,
                "root_causes": analysis.root_causes,
                "systemic_issues": analysis.systemic_issues,
                "prioritized_fixes": analysis.prioritized_fixes,
                "affected_components": analysis.affected_components,
                "individual_failures": [
                    {
                        "test_name": f.test_name,
                        "test_file": f.test_file,
                        "error_message": f.error_message,
                        "severity": f.severity,
                        "categorization_rationale": f.categorization_rationale,
                    }
                    for f in self.failures
                    if f.new_main_category == category
                ],
            }

        return report

    def _generate_cross_category_insights(self) -> dict[str, Any]:
        """Generate insights across all categories."""
        insights: dict[str, Any] = {
            "most_affected_category": max(
                self.category_analyses.keys(),
                key=lambda cat: self.category_analyses[cat].total_failures,
            ).value,
            "highest_severity_distribution": {},
            "common_themes": [],
        }

        # Severity distribution across categories
        severity_dist: dict[str, int] = {}
        for category, analysis in self.category_analyses.items():
            high_severity_count = analysis.severity_breakdown.get("high", 0)
            severity_dist[category.value] = high_severity_count
        insights["highest_severity_distribution"] = severity_dist

        # Common themes
        all_components: list[str] = []
        for analysis in self.category_analyses.values():
            all_components.extend(analysis.affected_components)

        # Find components affected by multiple categories
        component_counts: dict[str, int] = {}
        for component in all_components:
            current_count = component_counts.get(component, 0)
            component_counts[component] = current_count + 1

        cross_category_components = [
            comp for comp, count in component_counts.items() if count > 1
        ]

        insights["common_themes"] = [
            f"Component '{comp}' affected by multiple failure categories"
            for comp in cross_category_components
        ]

        return insights

    def save_report(self, output_path: str) -> None:
        """Save categorization report to JSON file."""
        report = self.generate_categorization_report()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Categorization report saved to: {output_path}")


def main() -> None:
    """Main execution function."""
    # Input and output paths
    input_file = (
        "artifacts/reports/comprehensive_failure_analysis/"
        "comprehensive_failures_20250713_063216.json"
    )
    output_file = "test_failure_categorization_report.json"

    # Create categorizer and process
    categorizer = TestFailureCategorizer(input_file)
    categorizer.load_and_categorize()

    # Generate and save report
    categorizer.save_report(output_file)

    # Print summary
    print("\n=== Test Failure Categorization Summary ===")
    print(f"Total failures analyzed: {len(categorizer.failures)}")
    print("\nCategory breakdown:")
    for category, analysis in categorizer.category_analyses.items():
        print(f"  {category.value}: {analysis.total_failures} failures")
        print(f"    Severity: {analysis.severity_breakdown}")
        top_pattern = (
            analysis.common_patterns[0] if analysis.common_patterns else "N/A"
        )
        print(f"    Top pattern: {top_pattern}")
        print()


if __name__ == "__main__":
    main()
