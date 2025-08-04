"""Coverage analysis functionality for monitoring system.

This module handles the core coverage analysis operations including
running coverage tests, parsing results, and extracting metrics.
"""

import json
import logging
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .config import CoverageMetrics

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Handles coverage analysis operations."""

    def __init__(self, target_threshold: float = 80.0) -> None:
        """Initialize the coverage analyzer.

        Args:
            target_threshold: Target coverage percentage
        """
        self.target_threshold = target_threshold

    def run_coverage_analysis(
        self, include_branch: bool = True, save_artifacts: bool = True
    ) -> CoverageMetrics:
        """Run comprehensive coverage analysis with multiple report formats.

        Args:
            include_branch: Include branch coverage analysis
            save_artifacts: Save HTML and XML reports

        Returns:
            CoverageMetrics object with complete analysis results
        """
        start_time = datetime.now()

        logger.info("Starting coverage analysis...")

        # Build pytest command with comprehensive reporting
        cmd = [
            "pytest",
            "--cov=src",
            "--cov=scripts/gui",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-report=xml:coverage.xml",
        ]

        if include_branch:
            cmd.append("--cov-branch")

        if save_artifacts:
            cmd.extend(
                [
                    "--cov-report=html:htmlcov",
                    "--html=outputs/coverage_monitoring/reports/test_report.html",
                    "--self-contained-html",
                ]
            )

        # Execute coverage analysis
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )

            if result.returncode not in [0, 1]:  # Allow test failures
                logger.warning(
                    "Coverage analysis completed with warnings: "
                    f"{result.returncode}"
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Coverage analysis failed: {e}")
            raise

        # Parse coverage results
        metrics = self._parse_coverage_results(
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        logger.info(
            f"Coverage analysis completed: {metrics.overall_coverage:.1f}%"
        )
        return metrics

    def _parse_coverage_results(
        self, execution_time: float
    ) -> CoverageMetrics:
        """Parse coverage results from generated files."""

        # Parse JSON coverage report
        coverage_data: dict[str, Any] = {}
        if Path("coverage.json").exists():
            with open("coverage.json") as f:
                coverage_data = json.load(f)

        # Parse XML for additional metadata
        branch_coverage: float | None = None
        if Path("coverage.xml").exists():
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            branch_rate = root.get("branch-rate")
            if branch_rate is not None:
                branch_coverage = float(branch_rate) * 100

        # Extract key metrics with type safety
        totals = cast(dict[str, Any], coverage_data.get("totals", {}))

        overall_coverage = cast(float, totals.get("percent_covered", 0.0))
        total_statements = cast(int, totals.get("num_statements", 0))
        covered_statements = cast(int, totals.get("covered_lines", 0))
        missing_statements = cast(int, totals.get("missing_lines", 0))

        # Analyze modules
        files = cast(dict[str, Any], coverage_data.get("files", {}))
        modules_count = len(files)
        modules_above_threshold = sum(
            1
            for file_data in files.values()
            if cast(
                float,
                cast(dict[str, Any], file_data.get("summary", {})).get(
                    "percent_covered", 0
                ),
            )
            >= self.target_threshold
        )

        # Identify critical gaps (modules with <50% coverage)
        critical_gaps = sum(
            1
            for file_data in files.values()
            if cast(
                float,
                cast(dict[str, Any], file_data.get("summary", {})).get(
                    "percent_covered", 0
                ),
            )
            < 50.0
        )

        return CoverageMetrics(
            timestamp=datetime.now().isoformat(),
            overall_coverage=overall_coverage,
            total_statements=total_statements,
            covered_statements=covered_statements,
            missing_statements=missing_statements,
            modules_count=modules_count,
            modules_above_threshold=modules_above_threshold,
            critical_gaps=critical_gaps,
            branch_coverage=branch_coverage,
            test_count=None,
            execution_time=execution_time,
        )

    def _get_git_commit_hash(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _get_git_branch(self) -> str | None:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
