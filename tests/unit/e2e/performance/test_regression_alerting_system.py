"""Unit tests for regression alerting system.

Tests the core functionality of performance regression detection, alert
generation, and notification systems with comprehensive coverage of edge
cases and error handling.
"""

import json
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tests.e2e.performance.regression_alerting_system import (
    NotificationChannel,
    NotificationConfig,
    RegressionAlert,
    RegressionAlertingSystem,
    RegressionAnalyzer,
    RegressionMetric,
    RegressionSeverity,
    RegressionThresholds,
    create_regression_alerting_system,
    process_ci_performance_results,
)


class TestRegressionThresholds:
    """Test regression thresholds configuration."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = RegressionThresholds()

        assert thresholds.warning_threshold == 15.0
        assert thresholds.critical_threshold == 25.0
        assert thresholds.min_samples == 3
        assert thresholds.confidence_level == 0.95
        assert thresholds.trend_window_hours == 24
        assert thresholds.sustained_degradation_minutes == 10

    def test_custom_thresholds(self) -> None:
        """Test custom threshold configuration."""
        thresholds = RegressionThresholds(
            warning_threshold=20.0, critical_threshold=35.0, min_samples=5
        )

        assert thresholds.warning_threshold == 20.0
        assert thresholds.critical_threshold == 35.0
        assert thresholds.min_samples == 5


class TestNotificationConfig:
    """Test notification configuration."""

    def test_default_config(self) -> None:
        """Test default notification configuration."""
        config = NotificationConfig()

        assert NotificationChannel.GITHUB_PR_COMMENT in config.enabled_channels
        assert config.github_token is None
        assert config.cooldown_minutes == 30

    def test_custom_config(self) -> None:
        """Test custom notification configuration."""
        config = NotificationConfig(
            enabled_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.WEBHOOK,
            ],
            github_token="test_token",
            cooldown_minutes=60,
        )

        assert len(config.enabled_channels) == 2
        assert NotificationChannel.EMAIL in config.enabled_channels
        assert config.github_token == "test_token"
        assert config.cooldown_minutes == 60


class TestRegressionAnalyzer:
    """Test regression analysis functionality."""

    @pytest.fixture
    def analyzer(self) -> RegressionAnalyzer:
        """Create regression analyzer instance."""
        thresholds = RegressionThresholds(
            warning_threshold=15.0, critical_threshold=25.0
        )
        return RegressionAnalyzer(thresholds)

    @pytest.fixture
    def sample_current_data(self) -> dict[str, Any]:
        """Sample current performance data."""
        return {
            "performance_gate_summary": {"total_violations": 8.0},
            "benchmark_results": {
                "smoke_test": {
                    "metrics": {
                        "average_response_time": 1200.0,
                        "peak_memory_mb": 6500.0,
                        "cpu_usage_percent": 85.0,
                    }
                }
            },
            "performance_violations": [
                {"type": "memory_threshold", "value": 6500},
                {"type": "response_time", "value": 1200},
                {"type": "memory_threshold", "value": 6600},
            ],
        }

    @pytest.fixture
    def sample_historical_data(self) -> list[dict[str, Any]]:
        """Sample historical performance data."""
        return [
            {
                "performance_gate_summary": {"total_violations": 2.0},
                "benchmark_results": {
                    "smoke_test": {
                        "metrics": {
                            "average_response_time": 800.0,
                            "peak_memory_mb": 5000.0,
                            "cpu_usage_percent": 60.0,
                        }
                    }
                },
            },
            {
                "performance_gate_summary": {"total_violations": 3.0},
                "benchmark_results": {
                    "smoke_test": {
                        "metrics": {
                            "average_response_time": 850.0,
                            "peak_memory_mb": 5200.0,
                            "cpu_usage_percent": 65.0,
                        }
                    }
                },
            },
            {
                "performance_gate_summary": {"total_violations": 1.0},
                "benchmark_results": {
                    "smoke_test": {
                        "metrics": {
                            "average_response_time": 780.0,
                            "peak_memory_mb": 4800.0,
                            "cpu_usage_percent": 58.0,
                        }
                    }
                },
            },
        ]

    def test_extract_metrics_complete_data(
        self, analyzer: RegressionAnalyzer, sample_current_data: dict[str, Any]
    ) -> None:
        """Test metric extraction from complete performance data."""
        metrics = analyzer._extract_metrics(sample_current_data)

        assert "violation_count" in metrics
        assert metrics["violation_count"] == 8.0
        assert "smoke_test_response_time" in metrics
        assert metrics["smoke_test_response_time"] == 1200.0
        assert "smoke_test_memory_usage" in metrics
        assert metrics["smoke_test_memory_usage"] == 6500.0
        assert "violations_memory_threshold" in metrics
        assert (
            metrics["violations_memory_threshold"] == 2.0
        )  # 2 memory violations

    def test_extract_metrics_partial_data(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test metric extraction from incomplete data."""
        partial_data = {"performance_gate_summary": {"total_violations": 5.0}}

        metrics = analyzer._extract_metrics(partial_data)
        assert "violation_count" in metrics
        assert metrics["violation_count"] == 5.0
        assert len(metrics) == 1  # Only violation count available

    def test_extract_metrics_empty_data(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test metric extraction from empty data."""
        metrics = analyzer._extract_metrics({})
        assert len(metrics) == 0

    def test_calculate_baselines(
        self,
        analyzer: RegressionAnalyzer,
        sample_historical_data: list[dict[str, Any]],
    ) -> None:
        """Test baseline calculation from historical data."""
        baselines = analyzer._calculate_baselines(sample_historical_data)

        assert "violation_count" in baselines
        assert baselines["violation_count"] == 2.0  # Median of [2, 3, 1]
        assert "smoke_test_response_time" in baselines
        assert (
            baselines["smoke_test_response_time"] == 800.0
        )  # Median of [800, 850, 780]

    def test_calculate_baselines_insufficient_data(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test baseline calculation with insufficient historical data."""
        insufficient_data = [
            {"performance_gate_summary": {"total_violations": 1.0}}
        ]
        baselines = analyzer._calculate_baselines(insufficient_data)

        assert len(baselines) == 0  # Not enough samples for reliable baseline

    def test_calculate_change_percentage(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test percentage change calculation."""
        # Normal case
        change = analyzer._calculate_change_percentage(120.0, 100.0)
        assert change == 20.0

        # Decrease case
        change = analyzer._calculate_change_percentage(80.0, 100.0)
        assert change == -20.0

        # Zero baseline case
        change = analyzer._calculate_change_percentage(50.0, 0.0)
        assert change == 100.0

        # Zero current with zero baseline
        change = analyzer._calculate_change_percentage(0.0, 0.0)
        assert change == 0.0

    def test_is_performance_degradation(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test performance degradation detection logic."""
        # Response time increase (degradation)
        assert (
            analyzer._is_performance_degradation("response_time", 20.0) is True
        )
        assert (
            analyzer._is_performance_degradation("response_time", 10.0)
            is False
        )

        # Memory usage increase (degradation)
        assert (
            analyzer._is_performance_degradation("memory_usage", 18.0) is True
        )

        # Violation count increase (degradation)
        assert (
            analyzer._is_performance_degradation("violation_count", 25.0)
            is True
        )

        # Success rate decrease (degradation for improvement metrics)
        assert (
            analyzer._is_performance_degradation("success_rate", -20.0) is True
        )
        assert (
            analyzer._is_performance_degradation("success_rate", 10.0) is False
        )

    def test_assess_regression_severity(
        self, analyzer: RegressionAnalyzer
    ) -> None:
        """Test regression severity assessment."""
        assert (
            analyzer._assess_regression_severity(10.0)
            == RegressionSeverity.LOW
        )
        assert (
            analyzer._assess_regression_severity(20.0)
            == RegressionSeverity.MEDIUM
        )
        assert (
            analyzer._assess_regression_severity(30.0)
            == RegressionSeverity.HIGH
        )
        assert (
            analyzer._assess_regression_severity(60.0)
            == RegressionSeverity.CRITICAL
        )

        # Test negative values (absolute value used)
        assert (
            analyzer._assess_regression_severity(-30.0)
            == RegressionSeverity.HIGH
        )

    def test_analyze_performance_data_with_regressions(
        self,
        analyzer: RegressionAnalyzer,
        sample_current_data: dict[str, Any],
        sample_historical_data: list[dict[str, Any]],
    ) -> None:
        """Test performance analysis detecting regressions."""
        regressions = analyzer.analyze_performance_data(
            sample_current_data, sample_historical_data
        )

        assert len(regressions) > 0

        # Check for violation count regression (8 vs baseline ~2)
        violation_regression = next(
            (r for r in regressions if r["metric_name"] == "violation_count"),
            None,
        )
        assert violation_regression is not None
        assert violation_regression["threshold_exceeded"] is True
        assert (
            violation_regression["change_percentage"] > 100
        )  # Significant increase

    def test_analyze_performance_data_insufficient_history(
        self, analyzer: RegressionAnalyzer, sample_current_data: dict[str, Any]
    ) -> None:
        """Test analysis with insufficient historical data."""
        insufficient_history = [
            {"performance_gate_summary": {"total_violations": 1.0}}
        ]

        regressions = analyzer.analyze_performance_data(
            sample_current_data, insufficient_history
        )

        assert len(regressions) == 0

    def test_analyze_performance_data_no_regressions(
        self,
        analyzer: RegressionAnalyzer,
        sample_historical_data: list[dict[str, Any]],
    ) -> None:
        """Test analysis with no regressions detected."""
        # Current data similar to historical baseline
        current_data = {
            "performance_gate_summary": {"total_violations": 2.0},
            "benchmark_results": {
                "smoke_test": {
                    "metrics": {
                        "average_response_time": 810.0,  # Close to baseline
                        "peak_memory_mb": 5100.0,
                        "cpu_usage_percent": 62.0,
                    }
                }
            },
        }

        regressions = analyzer.analyze_performance_data(
            current_data, sample_historical_data
        )
        assert len(regressions) == 0


class TestRegressionAlertingSystem:
    """Test main alerting system functionality."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def alerting_system(self, temp_dir: Path) -> RegressionAlertingSystem:
        """Create alerting system instance."""
        thresholds = RegressionThresholds()
        notification_config = NotificationConfig()

        return RegressionAlertingSystem(
            thresholds=thresholds,
            notification_config=notification_config,
            historical_data_path=temp_dir / "historical",
        )

    @pytest.fixture
    def sample_performance_results(self, temp_dir: Path) -> Path:
        """Create sample performance results file."""
        results_data = {
            "performance_gate_summary": {"total_violations": 10},
            "benchmark_results": {
                "test": {"metrics": {"average_response_time": 1500.0}}
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        results_file = temp_dir / "performance_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f)

        return results_file

    def test_initialization(self, temp_dir: Path) -> None:
        """Test alerting system initialization."""
        system = RegressionAlertingSystem(
            historical_data_path=temp_dir / "test"
        )

        assert system.thresholds is not None
        assert system.notification_config is not None
        assert system.historical_data_path.exists()
        assert system.analyzer is not None

    def test_store_historical_data(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test storing historical performance data."""
        test_data = {
            "performance_gate_summary": {"total_violations": 5},
            "test_key": "test_value",
        }

        alerting_system._store_historical_data(test_data, "abc123")

        # Check that file was created
        historical_files = list(
            alerting_system.historical_data_path.glob("performance_*.json")
        )
        assert len(historical_files) == 1

        # Check file content
        with open(historical_files[0]) as f:
            stored_data = json.load(f)

        assert stored_data["test_key"] == "test_value"
        assert stored_data["commit_sha"] == "abc123"
        assert "timestamp" in stored_data
        assert "stored_by" in stored_data

    def test_load_historical_data_empty(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test loading historical data when none exists."""
        historical_data = alerting_system._load_historical_data()
        assert len(historical_data) == 0

    def test_load_historical_data_with_files(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test loading existing historical data files."""
        # Create some test historical files
        for i in range(3):
            timestamp = (
                (datetime.now(UTC) - timedelta(days=i))
                .isoformat()
                .replace(":", "_")
            )
            file_path = (
                alerting_system.historical_data_path
                / f"performance_{timestamp}.json"
            )

            test_data = {
                "performance_gate_summary": {"total_violations": i + 1},
                "timestamp": timestamp,
            }

            with open(file_path, "w") as f:
                json.dump(test_data, f)

        historical_data = alerting_system._load_historical_data()
        assert len(historical_data) == 3

        # Should be sorted by timestamp (most recent first)
        violations = [
            d["performance_gate_summary"]["total_violations"]
            for d in historical_data
        ]
        assert violations == [1, 2, 3]  # Most recent (day 0) first

    def test_generate_alerts(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test alert generation from regression metrics."""
        regression_metrics: list[RegressionMetric] = [
            {
                "metric_name": "response_time",
                "current_value": 1200.0,
                "baseline_value": 800.0,
                "change_percentage": 50.0,
                "threshold_exceeded": True,
                "severity": "high",
            }
        ]

        current_data = {"test": "data"}
        alerts = alerting_system._generate_alerts(
            regression_metrics, current_data, "abc123"
        )

        assert len(alerts) == 1
        alert = alerts[0]

        assert alert["severity"] == "high"
        assert alert["metric_name"] == "response_time"
        assert alert["change_percentage"] == 50.0
        assert "abc123" in alert["message"]
        assert alert["context"]["commit_sha"] == "abc123"

    def test_generate_alerts_with_cooldown(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test alert generation respects cooldown periods."""
        regression_metrics: list[RegressionMetric] = [
            {
                "metric_name": "response_time",
                "current_value": 1200.0,
                "baseline_value": 800.0,
                "change_percentage": 50.0,
                "threshold_exceeded": True,
                "severity": "high",
            }
        ]

        # Generate first alert
        alerts1 = alerting_system._generate_alerts(
            regression_metrics, {}, "abc123"
        )
        assert len(alerts1) == 1

        # Generate second alert immediately (should be blocked by cooldown)
        alerts2 = alerting_system._generate_alerts(
            regression_metrics, {}, "def456"
        )
        assert len(alerts2) == 0

    def test_generate_alert_message(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test alert message generation."""
        regression: RegressionMetric = {
            "metric_name": "response_time",
            "current_value": 1200.0,
            "baseline_value": 800.0,
            "change_percentage": 50.0,
            "threshold_exceeded": True,
            "severity": "high",
        }

        message = alerting_system._generate_alert_message(regression, "abc123")

        assert "Performance Regression Detected" in message
        assert "response_time" in message
        assert "increased by 50.0%" in message
        assert "abc123" in message
        assert "HIGH" in message

    @patch.dict(
        "os.environ", {"GITHUB_RUN_NUMBER": "42", "GITHUB_REF_NAME": "main"}
    )
    def test_generate_alert_message_with_context(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test alert message includes CI context."""
        regression: RegressionMetric = {
            "metric_name": "memory_usage",
            "current_value": 6000.0,
            "baseline_value": 4000.0,
            "change_percentage": 50.0,
            "threshold_exceeded": True,
            "severity": "medium",
        }

        message = alerting_system._generate_alert_message(regression, "abc123")

        assert "Build: 42" in message
        assert "Branch: main" in message

    @patch("builtins.open", new_callable=mock_open)
    def test_send_github_pr_comment(
        self, mock_file: MagicMock, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test GitHub PR comment generation."""
        alert: RegressionAlert = {
            "alert_id": "test_alert",
            "severity": "high",
            "metric_name": "response_time",
            "change_percentage": 50.0,
            "current_value": 1200.0,
            "baseline_value": 800.0,
            "threshold": 15.0,
            "message": "Test regression alert message",
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {"test": "data"},
        }

        result = alerting_system._send_github_pr_comment(alert)

        assert result is True
        mock_file.assert_called_once_with(
            Path("regression-alert-comment.md"), "w"
        )

    @patch(
        "tests.e2e.performance.regression_alerting_system.RegressionAlertingSystem._send_github_pr_comment"
    )
    def test_send_alert_notifications(
        self, mock_github: MagicMock, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test alert notification sending."""
        mock_github.return_value = True

        alert: RegressionAlert = {
            "alert_id": "test_alert",
            "severity": "high",
            "metric_name": "response_time",
            "change_percentage": 50.0,
            "current_value": 1200.0,
            "baseline_value": 800.0,
            "threshold": 15.0,
            "message": "Test message",
            "timestamp": "2024-01-01T00:00:00Z",
            "context": {"test": "data"},
        }

        result = alerting_system._send_alert_notifications(alert)

        assert result["success"] is True
        assert len(result["channels_successful"]) == 1
        assert (
            NotificationChannel.GITHUB_PR_COMMENT.value
            in result["channels_successful"]
        )
        mock_github.assert_called_once()

    def test_process_performance_results_success(
        self,
        alerting_system: RegressionAlertingSystem,
        sample_performance_results: Path,
    ) -> None:
        """Test successful processing of performance results."""
        # Add some historical data first
        historical_data = {
            "performance_gate_summary": {"total_violations": 2},
            "timestamp": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }
        alerting_system._store_historical_data(historical_data)

        summary = alerting_system.process_performance_results(
            sample_performance_results, "abc123"
        )

        assert "error" not in summary
        assert "regressions_detected" in summary
        assert "alerts_generated" in summary
        assert summary["commit_sha"] == "abc123"

    def test_process_performance_results_file_not_found(
        self, alerting_system: RegressionAlertingSystem
    ) -> None:
        """Test processing with missing results file."""
        summary = alerting_system.process_performance_results(
            "nonexistent_file.json", "abc123"
        )

        assert "error" in summary
        assert "No such file" in summary[
            "error"
        ] or "FileNotFoundError" in str(summary["error"])


class TestConvenienceFunctions:
    """Test convenience functions for CI/CD integration."""

    @patch.dict(
        "os.environ",
        {
            "REGRESSION_WARNING_THRESHOLD": "20.0",
            "REGRESSION_CRITICAL_THRESHOLD": "35.0",
            "ALERT_COOLDOWN_MINUTES": "45",
        },
    )
    def test_create_regression_alerting_system_from_env(self) -> None:
        """Test creating alerting system from environment variables."""
        system = create_regression_alerting_system()

        assert system.thresholds.warning_threshold == 20.0
        assert system.thresholds.critical_threshold == 35.0
        assert system.notification_config.cooldown_minutes == 45

    @patch(
        "tests.e2e.performance.regression_alerting_system.create_regression_alerting_system"
    )
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"performance_gate_summary": {"total_violations": 5}}',
    )
    def test_process_ci_performance_results_success(
        self, mock_file: MagicMock, mock_create: MagicMock
    ) -> None:
        """Test CI performance results processing with no regressions."""
        mock_system = MagicMock()
        mock_system.process_performance_results.return_value = {
            "regressions_detected": 0,
            "alerts_generated": 0,
            "notifications_sent": 0,
        }
        mock_create.return_value = mock_system

        exit_code = process_ci_performance_results("test_results.json")

        assert exit_code == 0
        mock_create.assert_called_once()
        mock_system.process_performance_results.assert_called_once()

    @patch(
        "tests.e2e.performance.regression_alerting_system.create_regression_alerting_system"
    )
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"performance_gate_summary": {"total_violations": 5}}',
    )
    def test_process_ci_performance_results_with_regressions(
        self, mock_file: MagicMock, mock_create: MagicMock
    ) -> None:
        """Test CI performance results processing with regressions detected."""
        mock_system = MagicMock()
        mock_system.process_performance_results.return_value = {
            "regressions_detected": 3,
            "alerts_generated": 2,
            "notifications_sent": 1,
        }
        mock_create.return_value = mock_system

        exit_code = process_ci_performance_results("test_results.json")

        assert exit_code == 1

    @patch(
        "tests.e2e.performance.regression_alerting_system.create_regression_alerting_system"
    )
    def test_process_ci_performance_results_error(
        self, mock_create: MagicMock
    ) -> None:
        """Test CI performance results processing with error."""
        mock_system = MagicMock()
        mock_system.process_performance_results.return_value = {
            "error": "Test error message"
        }
        mock_create.return_value = mock_system

        exit_code = process_ci_performance_results("test_results.json")

        assert exit_code == 1

    @patch(
        "tests.e2e.performance.regression_alerting_system.create_regression_alerting_system"
    )
    def test_process_ci_performance_results_exception(
        self, mock_create: MagicMock
    ) -> None:
        """Test CI performance results processing with exception."""
        mock_create.side_effect = Exception("Test exception")

        exit_code = process_ci_performance_results("test_results.json")

        assert exit_code == 1


@pytest.mark.integration
class TestRegressionAlertingIntegration:
    """Integration tests for regression alerting system."""

    def test_end_to_end_regression_detection(self, tmp_path: Path) -> None:
        """Test complete end-to-end regression detection workflow."""
        # Setup alerting system
        system = RegressionAlertingSystem(
            historical_data_path=tmp_path / "historical"
        )

        # Create historical baseline data
        for i in range(5):
            historical_data = {
                "performance_gate_summary": {"total_violations": 2 + i % 2},
                "benchmark_results": {
                    "test": {
                        "metrics": {"average_response_time": 800.0 + (i * 10)}
                    }
                },
                "timestamp": (
                    datetime.now(UTC) - timedelta(days=i + 1)
                ).isoformat(),
            }
            system._store_historical_data(historical_data, f"commit_{i}")

        # Create current data showing regression
        current_data = {
            "performance_gate_summary": {
                "total_violations": 15
            },  # Significant increase
            "benchmark_results": {
                "test": {
                    "metrics": {"average_response_time": 1200.0}
                }  # 50% slower
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Save current data to file
        results_file = tmp_path / "current_results.json"
        with open(results_file, "w") as f:
            json.dump(current_data, f)

        # Process results
        summary = system.process_performance_results(
            results_file, "current_commit"
        )

        # Verify regression detection
        assert summary["regressions_detected"] > 0
        assert summary["alerts_generated"] > 0
        assert "error" not in summary

        # Verify historical data was stored
        historical_files = list(
            system.historical_data_path.glob("performance_*.json")
        )
        assert len(historical_files) == 6  # 5 baseline + 1 current
