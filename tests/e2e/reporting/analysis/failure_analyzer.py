"""Failure analysis module for test result classification.

This module provides comprehensive failure pattern recognition and analysis
capabilities for identifying common issues and generating actionable insights.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class FailurePattern(TypedDict):
    """Type definition for failure pattern detection."""

    pattern_id: str
    pattern_name: str
    error_regex: str
    frequency: int
    last_seen: str
    recommended_action: str
    severity: str


class FailureAnalyzer:
    """Analyzer for test failure patterns and classification.

    Provides automatic failure classification, pattern recognition,
    and actionable recommendations based on failure analysis.
    """

    def __init__(self, historical_data_path: Path | None = None) -> None:
        """Initialize the failure analyzer.

        Args:
            historical_data_path: Path to store historical failure data
        """
        self.historical_data_path = historical_data_path or Path(
            "test-reports/historical"
        )
        self.historical_data_path.mkdir(parents=True, exist_ok=True)

        # Predefined failure patterns for common issues
        self.failure_patterns = [
            {
                "pattern_id": "timeout_error",
                "pattern_name": "Timeout Error",
                "error_regex": r"timeout|TimeoutException|timed out",
                "recommended_action": (
                    "Check network connectivity and increase timeout values"
                ),
                "severity": "warning",
            },
            {
                "pattern_id": "element_not_found",
                "pattern_name": "Element Not Found",
                "error_regex": (
                    r"NoSuchElementException|element not found|"
                    r"could not find element"
                ),
                "recommended_action": (
                    "Verify element locators and page load timing"
                ),
                "severity": "error",
            },
            {
                "pattern_id": "stale_element",
                "pattern_name": "Stale Element Reference",
                "error_regex": r"StaleElementReferenceException|stale element",
                "recommended_action": "Re-find elements after page changes",
                "severity": "warning",
            },
            {
                "pattern_id": "connection_error",
                "pattern_name": "Connection Error",
                "error_regex": (
                    r"ConnectionError|connection refused|network error"
                ),
                "recommended_action": (
                    "Check Streamlit app startup and port availability"
                ),
                "severity": "critical",
            },
            {
                "pattern_id": "assert_error",
                "pattern_name": "Assertion Failure",
                "error_regex": r"AssertionError|assertion failed",
                "recommended_action": (
                    "Review test expectations and actual application behavior"
                ),
                "severity": "error",
            },
            {
                "pattern_id": "browser_crash",
                "pattern_name": "Browser Crash",
                "error_regex": (
                    r"browser.*crash|chrome.*crash|session.*deleted"
                ),
                "recommended_action": (
                    "Check system resources and browser stability"
                ),
                "severity": "critical",
            },
        ]

        logger.debug("Failure analyzer initialized")

    def analyze_failures(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze test failures and classify them into patterns.

        Args:
            test_results: List of test results to analyze

        Returns:
            Dictionary containing failure analysis results
        """
        failed_tests = [
            result
            for result in test_results
            if result["status"] in ["failed", "error"]
        ]

        if not failed_tests:
            return {
                "total_failures": 0,
                "failure_patterns": [],
                "unclassified_failures": [],
                "recommendations": [],
            }

        # Classify failures by patterns
        pattern_matches = self._classify_failures(failed_tests)

        # Generate recommendations
        recommendations = self._generate_recommendations(pattern_matches)

        # Find unclassified failures
        unclassified = self._find_unclassified_failures(
            failed_tests, pattern_matches
        )

        return {
            "total_failures": len(failed_tests),
            "failure_patterns": pattern_matches,
            "unclassified_failures": unclassified,
            "recommendations": recommendations,
            "failure_rate": len(failed_tests) / len(test_results) * 100,
        }

    def _classify_failures(
        self, failed_tests: list[dict[str, Any]]
    ) -> list[FailurePattern]:
        """Classify failures into known patterns."""
        pattern_matches = []

        for pattern in self.failure_patterns:
            matching_tests = []
            regex = re.compile(pattern["error_regex"], re.IGNORECASE)

            for test in failed_tests:
                error_text = (
                    (test.get("error_message") or "")
                    + " "
                    + (test.get("failure_reason") or "")
                )
                if regex.search(error_text):
                    matching_tests.append(test["test_name"])

            if matching_tests:
                pattern_match: FailurePattern = {
                    "pattern_id": pattern["pattern_id"],
                    "pattern_name": pattern["pattern_name"],
                    "error_regex": pattern["error_regex"],
                    "frequency": len(matching_tests),
                    "last_seen": datetime.now().isoformat(),
                    "recommended_action": pattern["recommended_action"],
                    "severity": pattern["severity"],
                }
                pattern_matches.append(pattern_match)

        return pattern_matches

    def _generate_recommendations(
        self, pattern_matches: list[FailurePattern]
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on failure patterns."""
        recommendations = []

        # Priority based on severity and frequency
        critical_patterns = [
            p for p in pattern_matches if p["severity"] == "critical"
        ]
        high_frequency_patterns = [
            p for p in pattern_matches if p["frequency"] >= 3
        ]

        if critical_patterns:
            recommendations.append(
                {
                    "priority": "high",
                    "title": "Critical Issues Detected",
                    "description": (
                        f"Found {len(critical_patterns)} critical failure "
                        f"patterns"
                    ),
                    "actions": [
                        p["recommended_action"] for p in critical_patterns
                    ],
                }
            )

        if high_frequency_patterns:
            recommendations.append(
                {
                    "priority": "medium",
                    "title": "Frequent Failure Patterns",
                    "description": (
                        f"Found {len(high_frequency_patterns)} patterns with "
                        f"high frequency"
                    ),
                    "actions": [
                        p["recommended_action"]
                        for p in high_frequency_patterns
                    ],
                }
            )

        return recommendations

    def _find_unclassified_failures(
        self,
        failed_tests: list[dict[str, Any]],
        pattern_matches: list[FailurePattern],
    ) -> list[dict[str, Any]]:
        """Find failures that don't match known patterns."""
        classified_tests = set()

        # Collect all tests that matched patterns
        for pattern in pattern_matches:
            regex = re.compile(pattern["error_regex"], re.IGNORECASE)
            for test in failed_tests:
                error_text = (
                    (test.get("error_message") or "")
                    + " "
                    + (test.get("failure_reason") or "")
                )
                if regex.search(error_text):
                    classified_tests.add(test["test_name"])

        # Find unclassified tests
        unclassified = []
        for test in failed_tests:
            if test["test_name"] not in classified_tests:
                unclassified.append(
                    {
                        "test_name": test["test_name"],
                        "error_message": test.get("error_message", ""),
                        "failure_reason": test.get("failure_reason", ""),
                    }
                )

        return unclassified

    def store_historical_data(self, analysis_results: dict[str, Any]) -> None:
        """Store failure analysis results for historical tracking."""
        filename = (
            self.historical_data_path / f"failure_analysis_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        logger.debug(f"Stored failure analysis data: {filename}")
