#!/usr/bin/env python3
"""
Test Coverage Validation Script - CrackSeg Project

This script provides automated validation of test coverage for the CrackSeg
project, ensuring that coverage meets the established 80% minimum threshold
and identifying specific areas that need attention.

Usage:
    python scripts/validate_coverage.py [options]

Options:
    --threshold FLOAT    Minimum coverage threshold (default: 80.0)
    --fail-under FLOAT   Fail if coverage is below this threshold
                         (default: 80.0)
    --generate-report    Generate detailed coverage report
    --output-dir PATH    Output directory for reports (default: outputs/)
    --verbose           Show detailed output
    --check-only        Only check coverage, don't run tests
    --html              Generate HTML coverage report
    --json              Generate JSON coverage report
    --badge             Generate coverage badge
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoverageValidator:
    """
    Validates test coverage and generates comprehensive reports.

    This class provides automated validation of test coverage with detailed
    analysis, reporting, and recommendations for improvement.
    """

    def __init__(
        self,
        threshold: float = 80.0,
        fail_under: float = 80.0,
        output_dir: Path = Path("outputs"),
        verbose: bool = False,
    ) -> None:
        """
        Initialize the coverage validator.

        Args:
            threshold: Minimum coverage threshold for reporting
            fail_under: Coverage threshold that causes validation to fail
            output_dir: Directory for output reports
            verbose: Enable verbose logging
        """
        self.threshold = threshold
        self.fail_under = fail_under
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Coverage data storage
        self.coverage_data: dict[str, Any] = {}
        self.module_coverage: dict[str, float] = {}
        self.critical_gaps: list[dict[str, Any]] = []

        logger.info(
            f"Coverage validator initialized with threshold: {threshold}%"
        )

    def run_coverage_tests(
        self, html: bool = True, json_report: bool = True, xml: bool = True
    ) -> bool:
        """
        Run pytest with coverage collection.

        Args:
            html: Generate HTML coverage report
            json_report: Generate JSON coverage report
            xml: Generate XML coverage report

        Returns:
            True if tests ran successfully, False otherwise
        """
        cmd = [
            "pytest",
            "--cov=src",
            "--cov-report=term-missing",
        ]

        if html:
            cmd.append("--cov-report=html")

        if json_report:
            cmd.append("--cov-report=json")

        if xml:
            cmd.append("--cov-report=xml")

        try:
            logger.info("Running coverage tests...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )

            if result.returncode == 0:
                logger.info("Coverage tests completed successfully")
                return True
            else:
                logger.warning(
                    f"Tests completed with warnings/failures: "
                    f"{result.returncode}"
                )
                return True  # Coverage still generated

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run coverage tests: {e}")
            return False
        except FileNotFoundError:
            logger.error(
                "pytest not found. Please install pytest and pytest-cov"
            )
            return False

    def load_coverage_data(self) -> bool:
        """
        Load coverage data from coverage.json file.

        Returns:
            True if data loaded successfully, False otherwise
        """
        coverage_file = Path("coverage.json")

        if not coverage_file.exists():
            logger.error(
                f"Coverage file not found: {coverage_file}. "
                f"Run with coverage first."
            )
            return False

        try:
            with open(coverage_file, encoding="utf-8") as f:
                self.coverage_data = json.load(f)

            logger.info("Coverage data loaded successfully")
            return True

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load coverage data: {e}")
            return False

    def analyze_coverage(self) -> dict[str, Any]:
        """
        Analyze coverage data and generate comprehensive analysis.

        Returns:
            Dictionary containing analysis results
        """
        if not self.coverage_data:
            logger.error("No coverage data available for analysis")
            return {}

        # Overall metrics
        totals = self.coverage_data["totals"]
        overall_coverage = totals["percent_covered"]
        statements_total = totals["num_statements"]
        statements_covered = totals["covered_lines"]
        statements_missing = totals["missing_lines"]

        # Module-level analysis
        files = self.coverage_data["files"]
        modules_above_threshold = 0
        modules_total = len(files)

        # Analyze each file
        for file_path, file_data in files.items():
            coverage_percent = file_data["summary"]["percent_covered"]
            self.module_coverage[file_path] = coverage_percent

            if coverage_percent >= self.threshold:
                modules_above_threshold += 1

            # Identify critical gaps
            if coverage_percent < self.threshold:
                gap_info = {
                    "file": file_path,
                    "coverage": coverage_percent,
                    "statements": file_data["summary"]["num_statements"],
                    "missing": file_data["summary"]["missing_lines"],
                    "missing_count": (
                        file_data["summary"]["missing_lines"]
                        if isinstance(
                            file_data["summary"]["missing_lines"], int
                        )
                        else len(file_data["summary"]["missing_lines"])
                    ),
                    "priority": self._calculate_priority(
                        file_path, coverage_percent, file_data
                    ),
                }
                self.critical_gaps.append(gap_info)

        # Sort critical gaps by priority and coverage
        self.critical_gaps.sort(
            key=lambda x: (
                {"P0": 0, "P1": 1, "P2": 2}[x["priority"]],
                x["coverage"],
            )
        )

        analysis = {
            "overall": {
                "coverage": overall_coverage,
                "statements_total": statements_total,
                "statements_covered": statements_covered,
                "statements_missing": statements_missing,
                "meets_threshold": overall_coverage >= self.threshold,
                "meets_fail_under": overall_coverage >= self.fail_under,
            },
            "modules": {
                "total_count": modules_total,
                "above_threshold": modules_above_threshold,
                "coverage_distribution": self._get_coverage_distribution(),
            },
            "critical_gaps": {
                "count": len(self.critical_gaps),
                "gaps": self.critical_gaps,
            },
            "recommendations": self._generate_recommendations(),
        }

        return analysis

    def _calculate_priority(
        self, file_path: str, coverage: float, file_data: dict[str, Any]
    ) -> str:
        """
        Calculate priority level for a file based on coverage and importance.

        Args:
            file_path: Path to the file
            coverage: Coverage percentage
            file_data: Coverage data for the file

        Returns:
            Priority level: P0 (critical), P1 (high), P2 (medium)
        """
        statements = file_data["summary"]["num_statements"]

        # P0: Critical files with very low coverage
        if coverage == 0 and statements > 10:
            return "P0"

        if (
            coverage < 20
            and statements > 50
            and any(
                critical in file_path.lower()
                for critical in ["main", "core", "model", "train"]
            )
        ):
            return "P0"

        # P1: Important files with low coverage
        if coverage < 50 and statements > 20:
            return "P1"

        # P2: Other files below threshold
        return "P2"

    def _get_coverage_distribution(self) -> dict[str, int]:
        """Get distribution of modules by coverage ranges."""
        distribution = {
            "excellent_90_100": 0,
            "good_80_89": 0,
            "fair_60_79": 0,
            "poor_40_59": 0,
            "critical_0_39": 0,
        }

        for coverage in self.module_coverage.values():
            if coverage >= 90:
                distribution["excellent_90_100"] += 1
            elif coverage >= 80:
                distribution["good_80_89"] += 1
            elif coverage >= 60:
                distribution["fair_60_79"] += 1
            elif coverage >= 40:
                distribution["poor_40_59"] += 1
            else:
                distribution["critical_0_39"] += 1

        return distribution

    def _generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        overall_coverage = (
            self.coverage_data["totals"]["percent_covered"]
            if self.coverage_data
            else 0
        )

        # Overall coverage recommendation
        if overall_coverage < self.fail_under:
            recommendations.append(
                {
                    "type": "critical",
                    "title": "Overall Coverage Below Threshold",
                    "description": (
                        f"Current coverage ({overall_coverage:.1f}%) is below "
                        f"the required threshold ({self.fail_under}%)"
                    ),
                    "action": (
                        "Implement comprehensive test suite following the "
                        "coverage improvement plan"
                    ),
                }
            )

        # Priority file recommendations
        p0_gaps = [
            gap for gap in self.critical_gaps if gap["priority"] == "P0"
        ]
        if p0_gaps:
            recommendations.append(
                {
                    "type": "critical",
                    "title": "Critical Priority Files Need Immediate "
                    "Attention",
                    "description": (
                        f"{len(p0_gaps)} P0 priority files have "
                        f"insufficient coverage"
                    ),
                    "action": (
                        f"Focus on: "
                        f"{', '.join([g['file'] for g in p0_gaps[:3]])}"
                    ),
                }
            )

        # Zero coverage files
        zero_coverage_files = [
            gap for gap in self.critical_gaps if gap["coverage"] == 0
        ]
        if zero_coverage_files:
            recommendations.append(
                {
                    "type": "high",
                    "title": "Files with Zero Coverage",
                    "description": (
                        f"{len(zero_coverage_files)} files have no test "
                        f"coverage"
                    ),
                    "action": (
                        "Implement basic unit tests for these files "
                        "immediately"
                    ),
                }
            )

        return recommendations

    def generate_detailed_report(self, analysis: dict[str, Any]) -> Path:
        """
        Generate a detailed coverage validation report.

        Args:
            analysis: Analysis results from analyze_coverage()

        Returns:
            Path to the generated report file
        """
        report_content = self._format_detailed_report(analysis)
        report_path = self.output_dir / "coverage_validation_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Detailed report generated: {report_path}")
        return report_path

    def _format_detailed_report(self, analysis: dict[str, Any]) -> str:
        """Format the detailed coverage report in Markdown."""
        overall = analysis["overall"]
        modules = analysis["modules"]

        report = f"""# Coverage Validation Report - CrackSeg

**Generated:** {self._get_timestamp()}
**Validation Threshold:** {self.threshold}%
**Failure Threshold:** {self.fail_under}%

## Summary

| Metric | Value | Status |
|--------|-------|--------|"""

        status_overall = "✅" if overall["meets_threshold"] else "❌"
        status_modules = (
            "✅"
            if modules["above_threshold"] == modules["total_count"]
            else "⚠️"
        )

        modules_text = f"{modules['above_threshold']}/{modules['total_count']}"

        report += f"""
| Overall Coverage | {overall["coverage"]:.1f}% | {status_overall} |
| Total Statements | {overall["statements_total"]:,} | - |
| Covered Statements | {overall["statements_covered"]:,} | - |
| Missing Statements | {overall["statements_missing"]:,} | - |
| Modules Above Threshold | {modules_text} | {status_modules} |

## Coverage Distribution

"""

        # Add distribution details
        distribution = modules["coverage_distribution"]
        for category, count in distribution.items():
            if count > 0:
                status_emoji = (
                    "✅"
                    if "excellent" in category or "good" in category
                    else "⚠️" if "fair" in category else "❌"
                )
                category_name = category.replace("_", " ").title()
                report += (
                    f"- {status_emoji} {category_name}: {count} modules\n"
                )

        report += "\n## Critical Coverage Gaps\n\n"

        if not self.critical_gaps:
            report += "🎉 No critical coverage gaps found!\n\n"
        else:
            report += (
                "| File | Coverage | Priority | Missing Lines | "
                "Action Required |\n"
            )
            report += (
                "|------|----------|----------|---------------|"
                "----------------|\n"
            )

            for gap in self.critical_gaps[:10]:  # Show top 10
                priority_emoji = (
                    "🚨"
                    if gap["priority"] == "P0"
                    else "⚠️" if gap["priority"] == "P1" else "📝"
                )
                action = (
                    "Immediate"
                    if gap["priority"] == "P0"
                    else "High" if gap["priority"] == "P1" else "Medium"
                )
                report += (
                    f"| `{gap['file']}` | {gap['coverage']:.1f}% | "
                    f"{priority_emoji} {gap['priority']} | "
                    f"{gap['missing_count']} | {action} |\n"
                )

        report += "\n## Recommendations\n\n"
        if overall["coverage"] < self.fail_under:
            report += (
                "1. **Immediate Action Required:** Coverage is below the "
                "failure threshold\n"
            )
            report += "2. Focus on P0 priority files first\n"
            report += (
                "3. Implement the coverage improvement plan systematically\n"
            )
        else:
            report += "1. Continue monitoring coverage on new code\n"
            report += "2. Address remaining gaps systematically\n"
            report += (
                "3. Consider increasing threshold for higher quality "
                "standards\n"
            )

        return report

    def generate_coverage_badge(self, analysis: dict[str, Any]) -> Path | None:
        """
        Generate a coverage badge for README.

        Args:
            analysis: Analysis results

        Returns:
            Path to badge file if successful, None otherwise
        """
        try:
            import requests

            coverage = analysis["overall"]["coverage"]
            color = (
                "red"
                if coverage < 50
                else "yellow" if coverage < 80 else "green"
            )

            badge_url = f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"

            response = requests.get(badge_url, timeout=10)
            response.raise_for_status()

            badge_path = self.output_dir / "coverage_badge.svg"
            with open(badge_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Coverage badge generated: {badge_path}")
            return badge_path

        except ImportError:
            logger.warning(
                "requests library not available, skipping badge generation"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to generate coverage badge: {e}")
            return None

    def validate_coverage(
        self,
        run_tests: bool = True,
        generate_reports: bool = True,
        html: bool = True,
        json_report: bool = True,
        badge: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Main validation workflow.

        Args:
            run_tests: Whether to run tests first
            generate_reports: Whether to generate detailed reports
            html: Generate HTML coverage report
            json_report: Generate JSON coverage report
            badge: Generate coverage badge

        Returns:
            Tuple of (validation_passed, analysis_results)
        """
        # Step 1: Run tests if requested
        if run_tests:
            if not self.run_coverage_tests(
                html=html, json_report=json_report, xml=False
            ):
                logger.error("Failed to run coverage tests")
                return False, {}

        # Step 2: Load coverage data
        if not self.load_coverage_data():
            return False, {}

        # Step 3: Analyze coverage
        analysis = self.analyze_coverage()
        if not analysis:
            return False, {}

        # Step 4: Determine validation result
        overall_coverage = analysis["overall"]["coverage"]
        validation_passed = overall_coverage >= self.fail_under

        # Step 5: Generate reports if requested
        if generate_reports:
            self.generate_detailed_report(analysis)

        if badge:
            self.generate_coverage_badge(analysis)

        # Step 6: Log results
        if validation_passed:
            logger.info(
                f"✅ Coverage validation PASSED: {overall_coverage:.1f}% >= "
                f"{self.fail_under}%"
            )
        else:
            logger.error(
                f"❌ Coverage validation FAILED: {overall_coverage:.1f}% < "
                f"{self.fail_under}%"
            )

        return validation_passed, analysis

    def _get_timestamp(self) -> str:
        """Get formatted timestamp for reports."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    """Main entry point for the coverage validation script."""
    parser = argparse.ArgumentParser(
        description="Validate test coverage for CrackSeg project"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Minimum coverage threshold for reporting (default: 80.0)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=80.0,
        help="Coverage threshold that causes validation to fail "
        "(default: 80.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for reports (default: outputs/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check existing coverage, don't run tests",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate detailed coverage report (default: True)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate HTML coverage report (default: True)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Generate JSON coverage report (default: True)",
    )
    parser.add_argument(
        "--badge",
        action="store_true",
        help="Generate coverage badge for README",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create validator
    validator = CoverageValidator(
        threshold=args.threshold,
        fail_under=args.fail_under,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Run validation
    validation_passed, analysis = validator.validate_coverage(
        run_tests=not args.check_only,
        generate_reports=args.generate_report,
        html=args.html,
        json_report=args.json,
        badge=args.badge,
    )

    # Output summary
    if analysis:
        overall = analysis["overall"]
        print(f"\n{'=' * 60}")
        print("COVERAGE VALIDATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Overall Coverage: {overall['coverage']:.1f}%")
        print(f"Threshold: {args.threshold}%")
        print(f"Status: {'PASS ✅' if validation_passed else 'FAIL ❌'}")
        print(
            f"Modules Above Threshold: "
            f"{analysis['modules']['above_threshold']}/"
            f"{analysis['modules']['total_count']}"
        )
        print(f"Critical Gaps: {analysis['critical_gaps']['count']}")

        if analysis["recommendations"]:
            print("\nTop Recommendations:")
            for i, rec in enumerate(analysis["recommendations"][:3], 1):
                print(f"{i}. {rec['title']}")

    return 0 if validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
