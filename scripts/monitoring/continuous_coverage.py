#!/usr/bin/env python3
"""
Continuous Coverage Automation Script This script provides automation
capabilities for continuous coverage monitoring, supporting multiple
modes of operation for CI/CD integration. Usage: # Run coverage
analysis and generate reports python
scripts/monitoring/continuous_coverage.py --mode analysis # Run full
monitoring with alerts python
scripts/monitoring/continuous_coverage.py --mode monitoring \
--threshold 80 # Check and send alerts only python
scripts/monitoring/continuous_coverage.py --mode alerts \\ --threshold
75 # Generate trend analysis for last 14 days python
scripts/monitoring/continuous_coverage.py --mode trends \\ --days 14 #
Combined operations for CI/CD python
scripts/monitoring/continuous_coverage.py --mode all \\ --threshold 80
--ci-mode
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for import s
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from crackseg.utils.monitoring.coverage import (
    AlertConfig,
    CoverageMetrics,
    CoverageMonitor,
)

# Type aliases for better readability
type ResultDict = dict[str, Any]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coverage_automation.log"),
    ],
)
logger = logging.getLogger(__name__)


class ContinuousCoverageAutomation:
    """
    Automation orchestrator for continuous coverage monitoring. This class
    coordinates all aspects of continuous coverage monitoring including
    analysis, alerting, trend tracking, and CI/CD integration.
    """

    def __init__(
        self,
        threshold: float = 80.0,
        output_dir: Path | None = None,
        ci_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the automation orchestrator. Args: threshold: Target
        coverage threshold percentage output_dir: Output directory for reports
        and data ci_mode: Enable CI/CD specific optimizations verbose: Enable
        verbose logging output
        """
        self.threshold = threshold
        self.ci_mode = ci_mode

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Set output directory based on mode
        if output_dir is None:
            if ci_mode:
                output_dir = Path("test-results/coverage_monitoring")
            else:
                output_dir = Path("outputs/coverage_monitoring")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize coverage monitor (will be set up later with alert config)
        self.monitor: CoverageMonitor | None = None

        logger.info(
            f"Continuous coverage automation initialized "
            f"(threshold: {threshold}%)"
        )

    def setup_monitor(self, alert_config: AlertConfig) -> None:
        """Setup coverage monitor with configuration."""
        self.monitor = CoverageMonitor(
            target_threshold=self.threshold,
            output_dir=self.output_dir,
            alert_config=alert_config,
        )

    def run_analysis_mode(self, include_artifacts: bool = True) -> ResultDict:
        """Run comprehensive coverage analysis.

        Args:
            include_artifacts: Generate HTML reports and artifacts

        Returns:
            Dictionary with analysis results and status
        """
        if not self.monitor:
            raise ValueError(
                "Monitor not initialized. Call setup_monitor() first."
            )

        logger.info("🔍 Starting comprehensive coverage analysis...")

        start_time = datetime.now()

        try:
            # Run coverage analysis
            metrics = self.monitor.run_coverage_analysis(
                include_branch=True, save_artifacts=include_artifacts
            )

            # Generate badge for README
            badge_path = self.monitor.generate_coverage_badge()

            # Check for alerts
            alerts_sent = self.monitor.check_and_send_alerts()

            execution_time = (datetime.now() - start_time).total_seconds()

            result: ResultDict = {
                "status": "success",
                "metrics": {
                    "coverage": metrics.overall_coverage,
                    "target": self.threshold,
                    "gap": self.threshold - metrics.overall_coverage,
                    "modules_total": metrics.modules_count,
                    "modules_above_threshold": metrics.modules_above_threshold,
                    "critical_gaps": metrics.critical_gaps,
                },
                "analysis": {
                    "execution_time": execution_time,
                    "timestamp": metrics.timestamp,
                    "branch_coverage": metrics.branch_coverage,
                },
                "alerts": {
                    "sent": alerts_sent,
                    "threshold_met": metrics.overall_coverage
                    >= self.threshold,
                },
                "artifacts": {
                    "badge_generated": (
                        badge_path.exists() if badge_path else False
                    ),
                    "reports_dir": str(self.output_dir / "reports"),
                },
            }

            # CI/CD specific output
            if self.ci_mode:
                self._generate_ci_outputs(result, metrics)

            logger.info(
                f"✅ Analysis completed: "
                f"{metrics.overall_coverage:.1f}% coverage"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def run_monitoring_mode(self, days_history: int = 30) -> ResultDict:
        """
        Run monitoring and trend analysis mode. Args: days_history: Number of
        days for trend analysis Returns: Dictionary with monitoring results
        """
        if not self.monitor:
            raise ValueError(
                "Monitor not initialized. Call setup_monitor() first."
            )

        logger.info(
            f"📊 Running monitoring mode (last {days_history} days)..."
        )

        try:
            # Analyze trends
            trend_data = self.monitor.analyze_trends(days=days_history)

            # Get current metrics if available
            current_metrics = None
            if self.monitor.current_metrics:
                current_metrics = {
                    "coverage": self.monitor.current_metrics.overall_coverage,
                    "timestamp": self.monitor.current_metrics.timestamp,
                }

            result: ResultDict = {
                "status": "success",
                "monitoring": {
                    "period_days": days_history,
                    "trend_analysis": trend_data,
                    "current_metrics": current_metrics,
                },
                "timestamp": datetime.now().isoformat(),
            }

            if self.ci_mode:
                self._generate_ci_monitoring_outputs(result)

            logger.info("✅ Monitoring analysis completed")
            return result

        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def run_alerts_mode(self) -> ResultDict:
        """
        Run alerts checking mode. Returns: Dictionary with alert results
        """
        if not self.monitor:
            raise ValueError(
                "Monitor not initialized. Call setup_monitor() first."
            )

        logger.info("🚨 Running alerts checking mode...")

        try:
            # Check if we have current metrics, if not run quick analysis
            if not self.monitor.current_metrics:
                logger.info(
                    "No current metrics available, running quick analysis..."
                )
                self.monitor.run_coverage_analysis(
                    include_branch=False, save_artifacts=False
                )

            # Check and send alerts
            alerts_sent = self.monitor.check_and_send_alerts()

            result: ResultDict = {
                "status": "success",
                "alerts": {
                    "sent": alerts_sent,
                    "current_coverage": (
                        self.monitor.current_metrics.overall_coverage
                        if self.monitor.current_metrics
                        else None
                    ),
                    "threshold": self.threshold,
                    "alert_config": {
                        "enabled": self.monitor.alert_config.enabled,
                        "warning_threshold": (
                            self.monitor.alert_config.threshold_warning
                        ),
                        "critical_threshold": (
                            self.monitor.alert_config.threshold_critical
                        ),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }

            if alerts_sent:
                logger.warning("⚠️ Coverage alerts were sent")
            else:
                logger.info("✅ No alerts needed")

            return result

        except Exception as e:
            logger.error(f"❌ Alert checking failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_ci_outputs(
        self, result: ResultDict, metrics: CoverageMetrics
    ) -> None:
        """Generate CI/CD specific outputs and artifacts."""

        # Generate environment variables file for subsequent jobs
        env_file = self.output_dir / "coverage.env"
        with open(env_file, "w") as f:
            f.write(f"COVERAGE_PERCENTAGE={metrics.overall_coverage:.1f}\n")
            f.write(f"COVERAGE_TARGET={self.threshold}\n")
            status = (
                "PASS"
                if metrics.overall_coverage >= self.threshold
                else "FAIL"
            )
            f.write(f"COVERAGE_STATUS={status}\n")
            f.write(f"CRITICAL_GAPS={metrics.critical_gaps}\n")
            f.write(
                f"MODULES_ABOVE_THRESHOLD={metrics.modules_above_threshold}\n"
            )
            f.write(f"TOTAL_MODULES={metrics.modules_count}\n")

        # Generate JSON summary for artifact upload
        summary_file = self.output_dir / "ci_summary.json"
        with open(summary_file, "w") as f:
            json.dump(result, f, indent=2)

        # Generate markdown summary for PR comments
        markdown_file = self.output_dir / "coverage_summary.md"
        self._generate_markdown_summary(markdown_file, result, metrics)

        logger.info("CI/CD artifacts generated successfully")

    def _generate_ci_monitoring_outputs(self, result: ResultDict) -> None:
        """Generate CI/CD specific monitoring outputs."""

        # Generate monitoring summary
        monitoring_file = self.output_dir / "monitoring_summary.json"
        with open(monitoring_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("CI/CD monitoring artifacts generated")

    def _generate_markdown_summary(
        self, output_path: Path, result: ResultDict, metrics: CoverageMetrics
    ) -> None:
        """Generate markdown summary for PR comments."""

        status_icon = "✅" if result["alerts"]["threshold_met"] else "❌"
        trend_icon = "📈" if result["metrics"]["gap"] <= 0 else "📉"

        # Format status indicators for table
        pass_fail = (
            "✅ PASS" if result["alerts"]["threshold_met"] else "❌ FAIL"
        )
        modules_ratio = (
            f"{metrics.modules_above_threshold}/{metrics.modules_count}"
        )
        gaps_status = "✅" if metrics.critical_gaps == 0 else "⚠️"

        coverage_row = (
            f"| **Overall Coverage** | {metrics.overall_coverage:.1f}% | "
            f"{self.threshold}% | {pass_fail} |"
        )

        markdown_content = f"""## {status_icon} Coverage Report

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
{coverage_row}
| **Modules Above Target** | {modules_ratio} | - | {trend_icon} |
| **Critical Gaps** | {metrics.critical_gaps} | 0 | {gaps_status} |

### Coverage Details
- **Total Statements**: {metrics.total_statements:,}
- **Covered Statements**: {metrics.covered_statements:,}
- **Missing Statements**: {metrics.missing_statements:,}
- **Branch Coverage**: {metrics.branch_coverage:.1f}% (if available)

### Analysis
- **Gap to Target**: {result["metrics"]["gap"]:.1f}%
- **Execution Time**: {result["analysis"]["execution_time"]:.2f}s
- **Timestamp**: {result["analysis"]["timestamp"]}

### Alerts
- **Alerts Sent**: {"Yes" if result["alerts"]["sent"] else "No"}
- **Threshold Met**: {"Yes" if result["alerts"]["threshold_met"] else "No"}

---
*Generated by CrackSeg Continuous Coverage Monitoring*
"""

        with open(output_path, "w") as f:
            f.write(markdown_content)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continuous coverage monitoring for CrackSeg project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["analysis", "monitoring", "alerts"],
        default="analysis",
        help="Operation mode (default: analysis)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Coverage threshold percentage (default: 80.0)",
    )

    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for reports and data"
    )

    parser.add_argument(
        "--ci-mode", action="store_true", help="Enable CI/CD integration mode"
    )

    parser.add_argument(
        "--email-recipients",
        type=str,
        help="Comma-separated list of email recipients for alerts",
    )

    parser.add_argument(
        "--slack-webhook", type=str, help="Slack webhook URL for alerts"
    )

    parser.add_argument(
        "--days-history",
        type=int,
        default=30,
        help="Days of history for trend analysis (default: 30)",
    )

    parser.add_argument(
        "--no-alerts", action="store_true", help="Disable alert sending"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for continuous coverage monitoring."""

    args = parse_arguments()

    try:
        # Setup alert configuration
        alert_config = AlertConfig(
            enabled=not args.no_alerts,
            email_recipients=(
                args.email_recipients.split(",")
                if args.email_recipients
                else []
            ),
            slack_webhook=args.slack_webhook,
            threshold_warning=args.threshold - 5.0,  # Warning 5% below target
            threshold_critical=args.threshold
            - 10.0,  # Critical 10% below target
            trend_alert_days=7,
            trend_decline_threshold=5.0,
        )

        # Initialize automation
        automation = ContinuousCoverageAutomation(
            threshold=args.threshold,
            output_dir=args.output_dir,
            ci_mode=args.ci_mode,
            verbose=args.verbose,
        )

        automation.setup_monitor(alert_config)

        # Run specified mode
        if args.mode == "analysis":
            result = automation.run_analysis_mode(
                include_artifacts=not args.ci_mode
            )
        elif args.mode == "monitoring":
            result = automation.run_monitoring_mode(
                days_history=args.days_history
            )
        elif args.mode == "alerts":
            result = automation.run_alerts_mode()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Print result summary
        print(json.dumps(result, indent=2))

        # Return appropriate exit code
        if result["status"] == "error":
            return 1
        elif args.mode == "analysis" and not result.get("alerts", {}).get(
            "threshold_met", True
        ):
            return 1  # Fail CI if coverage below threshold
        else:
            return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
