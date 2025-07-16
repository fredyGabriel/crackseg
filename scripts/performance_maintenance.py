#!/usr/bin/env python3
"""
Performance Benchmarking System Maintenance Script

This script automates maintenance tasks for the performance benchmarking
system:
- System health checks and validation
- Performance baseline updates
- Resource monitoring validation
- CI/CD integration verification
- Cleanup system validation

Usage:
    python scripts/performance_maintenance.py --health-check
    python scripts/performance_maintenance.py --update-baselines
    python scripts/performance_maintenance.py --validate-cleanup
    python scripts/performance_maintenance.py --full-maintenance
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_print(text: str) -> None:
    """
    Print text with Unicode-safe emoji handling for Windows.

    Args:
        text: Text to print, potentially containing emojis
    """
    # Define emoji replacements for Windows compatibility
    emoji_replacements = {
        "🔍": "[CHECK]",
        "❌": "[ERROR]",
        "✅": "[SUCCESS]",
        "⚠️": "[WARNING]",
        "💡": "[TIP]",
        "🔧": "[MAINTENANCE]",
        "📊": "[REPORT]",
        "🚀": "[STARTING]",
        "🏁": "[COMPLETE]",
    }

    # Replace emojis if encoding is problematic
    if sys.stdout.encoding in ["cp1252", "ascii"]:
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)

    print(text)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("performance_maintenance.log"),
    ],
)
logger = logging.getLogger(__name__)


class PerformanceMaintenanceManager:
    """Manages automated maintenance tasks for the performance benchmarking
    system."""

    def __init__(self) -> None:
        """Initialize the maintenance manager."""
        self.project_root = Path.cwd()
        self.performance_dir = (
            self.project_root / "tests" / "e2e" / "performance"
        )
        self.cleanup_dir = self.project_root / "tests" / "e2e" / "cleanup"
        self.config_dir = self.project_root / "configs" / "testing"
        self.maintenance_log: list[str] = []

    def log_action(
        self, action: str, status: str, details: str | None = None
    ) -> None:
        """Log maintenance action with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}: {status}"
        if details:
            log_entry += f" - {details}"

        self.maintenance_log.append(log_entry)
        logger.info(log_entry)

    def run_command(
        self, command: list[str], description: str
    ) -> tuple[bool, str]:
        """Execute shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                self.log_action(description, "SUCCESS", result.stdout.strip())
                return True, result.stdout.strip()
            else:
                self.log_action(description, "FAILED", result.stderr.strip())
                return False, result.stderr.strip()

        except subprocess.TimeoutExpired:
            self.log_action(
                description, "TIMEOUT", "Command exceeded 5 minute timeout"
            )
            return False, "Command timeout"
        except Exception as e:
            self.log_action(description, "ERROR", str(e))
            return False, str(e)

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive system health check."""
        logger.info("Starting performance system health check...")

        health_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_status": "UNKNOWN",
        }

        # Check 1: Validate performance system structure
        success, output = self.run_command(
            ["python", "scripts/check_test_files.py", "--performance-check"],
            "Performance system validation",
        )
        health_results["checks"]["system_validation"] = {
            "status": "PASS" if success else "FAIL",
            "output": output,
        }

        # Check 2: Validate performance thresholds configuration
        thresholds_file = self.config_dir / "performance_thresholds.yaml"
        if thresholds_file.exists():
            self.log_action(
                "Thresholds configuration", "FOUND", str(thresholds_file)
            )
            health_results["checks"]["thresholds_config"] = {"status": "PASS"}
        else:
            self.log_action(
                "Thresholds configuration", "MISSING", str(thresholds_file)
            )
            health_results["checks"]["thresholds_config"] = {"status": "FAIL"}

        # Check 3: Validate CI/CD integration
        ci_workflow = (
            self.project_root / ".github" / "workflows" / "performance-ci.yml"
        )
        if ci_workflow.exists():
            self.log_action("CI/CD workflow", "FOUND", str(ci_workflow))
            health_results["checks"]["ci_integration"] = {"status": "PASS"}
        else:
            self.log_action("CI/CD workflow", "MISSING", str(ci_workflow))
            health_results["checks"]["ci_integration"] = {"status": "FAIL"}

        # Check 4: Test resource monitoring
        success, output = self.run_command(
            [
                "python",
                "-c",
                "from crackseg.utils.monitoring.resource_monitor import "
                "ResourceMonitor; print('OK')",
            ],
            "Resource monitoring import test",
        )
        health_results["checks"]["resource_monitoring"] = {
            "status": "PASS" if success else "FAIL",
            "output": output,
        }

        # Check 5: Test cleanup validation
        if self.cleanup_dir.exists():
            success, output = self.run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    str(self.cleanup_dir),
                    "-k",
                    "validation",
                    "--collect-only",
                ],
                "Cleanup validation test discovery",
            )
            health_results["checks"]["cleanup_validation"] = {
                "status": "PASS" if success else "FAIL",
                "output": output,
            }
        else:
            health_results["checks"]["cleanup_validation"] = {"status": "FAIL"}

        # Determine overall status
        failed_checks = [
            check
            for check, result in health_results["checks"].items()
            if result["status"] == "FAIL"
        ]

        if not failed_checks:
            health_results["overall_status"] = "HEALTHY"
            logger.info("All health checks passed")
        else:
            health_results["overall_status"] = "UNHEALTHY"
            logger.warning(
                f"{len(failed_checks)} health checks failed: {failed_checks}"
            )

        return health_results

    def update_baselines(self) -> dict[str, Any]:
        """Update performance baselines based on recent runs."""
        logger.info("Updating performance baselines...")

        update_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "updated_baselines": [],
            "status": "UNKNOWN",
        }

        # Run baseline generation
        success, output = self.run_command(
            [
                "python",
                "-m",
                "pytest",
                str(self.performance_dir),
                "--generate-baseline",
                "-v",
            ],
            "Performance baseline generation",
        )

        if success:
            update_results["status"] = "SUCCESS"
            update_results["updated_baselines"] = ["performance_benchmarks"]
            logger.info("Performance baselines updated successfully")
        else:
            update_results["status"] = "FAILED"
            logger.error("Failed to update performance baselines")

        return update_results

    def validate_cleanup_system(self) -> dict[str, Any]:
        """Validate the cleanup system functionality."""
        logger.info("Validating cleanup system...")

        validation_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "status": "UNKNOWN",
        }

        # Test cleanup validation components
        cleanup_tests = [
            ("cleanup_manager", "test_cleanup_manager"),
            ("validation_system", "test_validation_system"),
            ("environment_readiness", "test_environment_readiness"),
        ]

        all_passed = True
        for component, test_pattern in cleanup_tests:
            success, output = self.run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    str(self.cleanup_dir),
                    "-k",
                    test_pattern,
                    "-v",
                ],
                f"Cleanup {component} validation",
            )

            validation_results["validation_tests"][component] = {
                "status": "PASS" if success else "FAIL",
                "output": output,
            }

            if not success:
                all_passed = False

        validation_results["status"] = "SUCCESS" if all_passed else "FAILED"

        if all_passed:
            logger.info("Cleanup system validation passed")
        else:
            logger.warning("Cleanup system validation failed")

        return validation_results

    def full_maintenance(self) -> dict[str, Any]:
        """Run complete maintenance cycle."""
        logger.info("Starting full maintenance cycle...")

        maintenance_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "maintenance_tasks": {},
            "overall_status": "UNKNOWN",
        }

        # Task 1: Health check
        health_results = self.health_check()
        maintenance_results["maintenance_tasks"][
            "health_check"
        ] = health_results

        # Task 2: Update baselines (only if health check passes)
        if health_results["overall_status"] == "HEALTHY":
            baseline_results = self.update_baselines()
            maintenance_results["maintenance_tasks"][
                "baseline_update"
            ] = baseline_results
        else:
            self.log_action(
                "Baseline update", "SKIPPED", "Health check failed"
            )

        # Task 3: Validate cleanup system
        cleanup_results = self.validate_cleanup_system()
        maintenance_results["maintenance_tasks"][
            "cleanup_validation"
        ] = cleanup_results

        # Task 4: Generate maintenance report
        self.generate_maintenance_report(maintenance_results)

        # Determine overall status
        failed_tasks = []
        for task, result in maintenance_results["maintenance_tasks"].items():
            if (
                result.get("status") == "FAILED"
                or result.get("overall_status") == "UNHEALTHY"
            ):
                failed_tasks.append(task)

        if not failed_tasks:
            maintenance_results["overall_status"] = "SUCCESS"
            logger.info("Full maintenance cycle completed successfully")
        else:
            maintenance_results["overall_status"] = "FAILED"
            logger.warning(
                f"Maintenance cycle failed. Failed tasks: {failed_tasks}"
            )

        return maintenance_results

    def generate_maintenance_report(self, results: dict[str, Any]) -> None:
        """Generate maintenance report."""
        report_file = self.project_root / "performance_maintenance_report.md"

        with open(report_file, "w") as f:
            f.write("# Performance System Maintenance Report\n\n")
            f.write(f"**Generated**: {results['timestamp']}\n\n")
            f.write(f"**Overall Status**: {results['overall_status']}\n\n")

            f.write("## Maintenance Tasks\n\n")
            for task, result in results["maintenance_tasks"].items():
                f.write(f"### {task.replace('_', ' ').title()}\n")
                f.write(f"**Status**: {result.get('status', 'N/A')}\n\n")

                if "checks" in result:
                    f.write("#### Checks\n")
                    for check, check_result in result["checks"].items():
                        status_emoji = (
                            "[PASS]"
                            if check_result["status"] == "PASS"
                            else "[FAIL]"
                        )
                        f.write(
                            f"- {status_emoji} {check}: "
                            f"{check_result['status']}\n"
                        )
                    f.write("\n")

            f.write("## Maintenance Log\n\n")
            for log_entry in self.maintenance_log:
                f.write(f"- {log_entry}\n")

            f.write("\n## Next Steps\n\n")
            if results["overall_status"] == "SUCCESS":
                f.write("- All maintenance tasks completed successfully\n")
                f.write("- Continue regular monitoring\n")
            else:
                f.write("- Review failed tasks and address issues\n")
                f.write("- Re-run maintenance after fixes\n")

        logger.info(f"Maintenance report generated: {report_file}")


def main() -> int:
    """Main entry point for the maintenance script."""
    parser = argparse.ArgumentParser(
        description="Performance Benchmarking System Maintenance Script"
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Run system health check"
    )
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update performance baselines",
    )
    parser.add_argument(
        "--validate-cleanup",
        action="store_true",
        help="Validate cleanup system",
    )
    parser.add_argument(
        "--full-maintenance",
        action="store_true",
        help="Run complete maintenance cycle",
    )

    args = parser.parse_args()

    # Initialize maintenance manager
    maintenance_manager = PerformanceMaintenanceManager()

    try:
        if args.health_check:
            results = maintenance_manager.health_check()
            return 0 if results["overall_status"] == "HEALTHY" else 1

        elif args.update_baselines:
            results = maintenance_manager.update_baselines()
            return 0 if results["status"] == "SUCCESS" else 1

        elif args.validate_cleanup:
            results = maintenance_manager.validate_cleanup_system()
            return 0 if results["status"] == "SUCCESS" else 1

        elif args.full_maintenance:
            results = maintenance_manager.full_maintenance()
            return 0 if results["overall_status"] == "SUCCESS" else 1

        else:
            parser.print_help()
            return 1

    except Exception as e:
        logger.error(f"Maintenance script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
