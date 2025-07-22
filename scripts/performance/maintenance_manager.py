"""Main performance maintenance manager."""

import argparse
import sys
from datetime import datetime
from typing import Any

from .baseline_updater import BaselineUpdater
from .cleanup_validator import CleanupValidator
from .health_checker import HealthChecker
from .utils import get_project_paths, setup_logging


class PerformanceMaintenanceManager:
    """Orchestrates all performance maintenance operations.

    Coordinates health checks, baseline updates, cleanup validation,
    and report generation.
    """

    def __init__(self) -> None:
        """Initialize the performance maintenance manager."""
        self.paths = get_project_paths()
        self.logger = setup_logging()

        # Initialize specialized components
        self.health_checker = HealthChecker(self.paths, self.logger)
        self.baseline_updater = BaselineUpdater(self.paths, self.logger)
        self.cleanup_validator = CleanupValidator(self.paths, self.logger)

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive system health check.

        Returns:
            Dictionary containing health check results
        """
        return self.health_checker.health_check()

    def update_baselines(self) -> dict[str, Any]:
        """Update performance baselines.

        Returns:
            Dictionary containing baseline update results
        """
        return self.baseline_updater.update_baselines()

    def validate_cleanup_system(self) -> dict[str, Any]:
        """Validate cleanup system functionality.

        Returns:
            Dictionary containing cleanup validation results
        """
        return self.cleanup_validator.validate_cleanup_system()

    def full_maintenance(self) -> dict[str, Any]:
        """Run complete maintenance cycle.

        Returns:
            Dictionary containing full maintenance results
        """
        self.logger.info("Starting full maintenance cycle...")

        maintenance_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "operations": {},
            "warnings": [],
            "errors": [],
        }

        # 1. Health Check
        try:
            self.logger.info("Running health check...")
            health_result = self.health_check()
            maintenance_results["operations"]["health_check"] = health_result

            if health_result.get("overall_status") == "error":
                maintenance_results["errors"].append(
                    "Health check failed - see health_check results"
                )
        except Exception as e:
            maintenance_results["errors"].append(f"Health check failed: {e}")

        # 2. Update Baselines
        try:
            self.logger.info("Updating performance baselines...")
            baseline_result = self.update_baselines()
            maintenance_results["operations"][
                "baseline_update"
            ] = baseline_result

            if baseline_result.get("overall_status") == "error":
                maintenance_results["errors"].append(
                    "Baseline update failed - see baseline_update results"
                )
        except Exception as e:
            maintenance_results["errors"].append(
                f"Baseline update failed: {e}"
            )

        # 3. Validate Cleanup System
        try:
            self.logger.info("Validating cleanup system...")
            cleanup_result = self.validate_cleanup_system()
            maintenance_results["operations"][
                "cleanup_validation"
            ] = cleanup_result

            if cleanup_result.get("overall_status") == "error":
                maintenance_results["errors"].append(
                    "Cleanup validation failed - see cleanup_validation results"  # noqa: E501
                )
        except Exception as e:
            maintenance_results["errors"].append(
                f"Cleanup validation failed: {e}"
            )

        # Determine overall status
        if maintenance_results["errors"]:
            maintenance_results["overall_status"] = "error"
        elif maintenance_results["warnings"]:
            maintenance_results["overall_status"] = "warning"
        else:
            maintenance_results["overall_status"] = "success"

        self.logger.info(
            f"Maintenance cycle completed with status: "
            f"{maintenance_results['overall_status']}"
        )
        return maintenance_results

    def generate_maintenance_report(self, results: dict[str, Any]) -> None:
        """Generate comprehensive maintenance report.

        Args:
            results: Results from maintenance operations
        """
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        from pathlib import Path

        artifacts_path = Path(self.paths.get("artifacts", "artifacts"))
        report_path = (
            artifacts_path
            / "maintenance"
            / f"maintenance_report_{report_timestamp}.txt"
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("PERFORMANCE MAINTENANCE REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            status = results.get("overall_status", "unknown")
            f.write(f"Overall Status: {status}\n\n")

            # Operations summary
            f.write("OPERATIONS SUMMARY\n")
            f.write("-" * 20 + "\n")
            for op_name, op_result in results.get("operations", {}).items():
                status = op_result.get("overall_status", "unknown")
                f.write(f"{op_name}: {status}\n")

            f.write("\n")

            # Errors and warnings
            if results.get("errors"):
                f.write("ERRORS\n")
                f.write("-" * 10 + "\n")
                for error in results["errors"]:
                    f.write(f"• {error}\n")
                f.write("\n")

            if results.get("warnings"):
                f.write("WARNINGS\n")
                f.write("-" * 10 + "\n")
                for warning in results["warnings"]:
                    f.write(f"• {warning}\n")
                f.write("\n")

            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            for op_name, op_result in results.get("operations", {}).items():
                f.write(f"\n{op_name.upper()}\n")
                f.write("=" * len(op_name) + "\n")
                f.write(str(op_result))
                f.write("\n")

        self.logger.info(f"Maintenance report generated: {report_path}")


def main() -> None:
    """Main entry point for the maintenance manager CLI."""
    parser = argparse.ArgumentParser(
        description="Performance maintenance manager for CrackSeg project"
    )
    parser.add_argument(
        "--operation",
        choices=["health", "baselines", "cleanup", "full"],
        default="full",
        help="Maintenance operation to perform",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed report after operation",
    )

    args = parser.parse_args()

    try:
        manager = PerformanceMaintenanceManager()

        if args.operation == "health":
            results = manager.health_check()
        elif args.operation == "baselines":
            results = manager.update_baselines()
        elif args.operation == "cleanup":
            results = manager.validate_cleanup_system()
        else:  # full
            results = manager.full_maintenance()

        # Print summary to console
        print(f"Operation '{args.operation}' completed")
        print(f"Status: {results.get('overall_status', 'unknown')}")

        if results.get("errors"):
            print(f"Errors: {len(results['errors'])}")
            for error in results["errors"][:3]:  # Show first 3 errors
                print(f"  • {error}")
            if len(results["errors"]) > 3:
                print(f"  ... and {len(results['errors']) - 3} more")

        if results.get("warnings"):
            print(f"Warnings: {len(results['warnings'])}")

        # Generate report if requested
        if args.report:
            manager.generate_maintenance_report(results)

        # Exit with appropriate code
        if results.get("overall_status") == "error":
            sys.exit(1)
        elif results.get("overall_status") == "warning":
            sys.exit(2)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Maintenance manager failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
