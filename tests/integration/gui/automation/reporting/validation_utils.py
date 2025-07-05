"""Validation utilities for integration test reporting.

This module provides validation functions for reporting configuration,
output directories, and system preconditions.
"""

from pathlib import Path
from typing import Any

from .stakeholder_reporting import StakeholderReportConfig


class ReportingValidationUtils:
    """Utilities for validating reporting configuration and preconditions."""

    @staticmethod
    def validate_reporting_configuration(
        config: StakeholderReportConfig,
    ) -> bool:
        """Validate reporting configuration.

        Args:
            config: Stakeholder report configuration

        Returns:
            True if configuration is valid
        """
        return len(config.export_formats) > 0 and (
            config.executive_enabled
            or config.technical_enabled
            or config.operations_enabled
        )

    @staticmethod
    def validate_output_directories() -> bool:
        """Validate output directories are accessible.

        Returns:
            True if directories can be created/accessed
        """
        try:
            # Test creating output directory
            test_dir = Path("comprehensive_reports")
            test_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = test_dir / "test_write.tmp"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()

            return True
        except Exception:
            return False

    @staticmethod
    def extract_reporting_errors(
        stakeholder_reports: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Extract reporting errors from stakeholder reports.

        Args:
            stakeholder_reports: Stakeholder reports

        Returns:
            List of error messages
        """
        errors = []
        for stakeholder, report in stakeholder_reports.items():
            if not report.get("success", False):
                errors.append(f"Failed to generate {stakeholder} report")
        return errors
