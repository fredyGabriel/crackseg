"""Test module for core performance report generator."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.e2e.performance.reporting.config import ReportConfiguration
from tests.e2e.performance.reporting.core import PerformanceReportGenerator


class TestPerformanceReportGenerator:
    """Test cases for PerformanceReportGenerator class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_path:
            return Path(temp_path)

    @pytest.fixture
    def sample_benchmark_results(self) -> dict[str, Any]:
        """Create sample benchmark results for testing."""
        return {
            "suite_name": "test_suite",
            "benchmark_results": {
                "load_test": {
                    "success_rate": 95.0,
                    "throughput": 150.0,
                    "metrics": {"response_time": 250.0},
                    "threshold_violations": [],
                },
                "stress_test": {
                    "success_rate": 88.0,
                    "throughput": 120.0,
                    "metrics": {"response_time": 400.0},
                    "threshold_violations": ["cpu_warning"],
                },
            },
        }

    @pytest.fixture
    def report_generator(self, temp_dir: Path) -> PerformanceReportGenerator:
        """Create report generator instance for testing."""
        config = ReportConfiguration()
        return PerformanceReportGenerator(
            storage_path=str(temp_dir / "reports"),
            historical_data_path=str(temp_dir / "historical"),
            config=config,
        )

    def test_initialization(self, temp_dir: Path) -> None:
        """Test report generator initialization."""
        config = ReportConfiguration()
        generator = PerformanceReportGenerator(
            storage_path=str(temp_dir / "reports"),
            historical_data_path=str(temp_dir / "historical"),
            config=config,
        )

        assert generator.storage_path == temp_dir / "reports"
        assert generator.historical_data_path == temp_dir / "historical"
        assert generator.config == config
        assert generator.storage_path.exists()
        assert generator.historical_data_path.exists()

    def test_initialization_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        generator = PerformanceReportGenerator()

        assert generator.storage_path == Path("performance-reports")
        assert generator.historical_data_path == Path(
            "performance-historical-data"
        )
        assert isinstance(generator.config, ReportConfiguration)

    def test_generate_comprehensive_report_basic(
        self,
        report_generator: PerformanceReportGenerator,
        sample_benchmark_results: dict[str, Any],
    ) -> None:
        """Test basic comprehensive report generation."""
        # Mock the existing instances created in __init__
        with (
            patch.object(
                report_generator.data_processor, "process_benchmark_results"
            ) as mock_process_data,
            patch.object(
                report_generator.historical_manager, "load_historical_data"
            ) as mock_load_historical,
            patch.object(
                report_generator.historical_manager, "store_current_data"
            ) as mock_store_data,
            patch.object(
                report_generator.historical_manager,
                "summarize_historical_data",
            ) as mock_summarize_historical,
            patch.object(
                report_generator.historical_manager, "get_data_file_info"
            ) as mock_get_file_info,
            patch.object(
                report_generator.analyzer, "perform_trend_analysis"
            ) as mock_trend,
            patch.object(
                report_generator.analyzer, "perform_regression_analysis"
            ) as mock_regression,
            patch.object(
                report_generator.analyzer, "generate_insights"
            ) as mock_insights,
            patch.object(
                report_generator.visualizer,
                "create_performance_visualizations",
            ) as mock_visualizations,
            patch.object(
                report_generator.formatter, "generate_html_dashboard"
            ) as mock_html,
            patch.object(
                report_generator.formatter, "generate_json_report"
            ) as mock_json,
        ):
            # Setup mock returns
            mock_process_data.return_value = {
                "overall_summary": {"average_success_rate": 91.5},
                "benchmark_details": {"load_test": {"success_rate": 95.0}},
            }
            mock_load_historical.return_value = []
            mock_store_data.return_value = None
            mock_summarize_historical.return_value = {"data_points": 0}
            mock_get_file_info.return_value = {"files": 0}
            mock_trend.return_value = {"trends": []}
            mock_regression.return_value = {"regressions_detected": 0}
            mock_insights.return_value = {"summary": "Test insights"}
            mock_visualizations.return_value = {"chart1": "data"}
            mock_html.return_value = Path("test.html")
            mock_json.return_value = Path("test.json")

            result = report_generator.generate_comprehensive_report(
                sample_benchmark_results, "test_commit"
            )

            assert isinstance(result, dict)
            assert "html" in result or "json" in result
            mock_process_data.assert_called_once_with(sample_benchmark_results)

    def test_get_storage_info(
        self, report_generator: PerformanceReportGenerator
    ) -> None:
        """Test storage information retrieval."""
        info = report_generator.get_storage_info()

        assert "storage_path" in info
        assert "historical_data_path" in info
        assert "storage_exists" in info
        assert "historical_exists" in info
        assert isinstance(info["storage_exists"], bool)
        assert isinstance(info["historical_exists"], bool)

    def test_validate_configuration_valid(
        self, report_generator: PerformanceReportGenerator
    ) -> None:
        """Test configuration validation with valid settings."""
        validation = report_generator.validate_configuration()

        assert validation["is_valid"] is True
        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["errors"], list)

    def test_validate_configuration_invalid_format(
        self, temp_dir: Path
    ) -> None:
        """Test configuration validation with invalid export format."""
        config = ReportConfiguration()
        config.export_formats = ["invalid_format"]

        generator = PerformanceReportGenerator(
            storage_path=str(temp_dir / "reports"),
            historical_data_path=str(temp_dir / "historical"),
            config=config,
        )

        validation = generator.validate_configuration()

        assert validation["is_valid"] is False
        assert len(validation["errors"]) > 0
        assert "invalid_format" in str(validation["errors"])

    def test_validate_configuration_invalid_time_window(
        self, temp_dir: Path
    ) -> None:
        """Test configuration validation with invalid time window."""
        config = ReportConfiguration()
        config.time_window_hours = -1

        generator = PerformanceReportGenerator(
            storage_path=str(temp_dir / "reports"),
            historical_data_path=str(temp_dir / "historical"),
            config=config,
        )

        validation = generator.validate_configuration()

        assert validation["is_valid"] is False
        assert any("Time window" in error for error in validation["errors"])

    def test_cleanup_old_data(
        self, report_generator: PerformanceReportGenerator
    ) -> None:
        """Test cleanup of old historical data."""
        with patch.object(
            report_generator.historical_manager, "cleanup_old_data"
        ) as mock_cleanup:
            mock_cleanup.return_value = 5

            result = report_generator.cleanup_old_data(max_files=100)

            assert result == 5
            mock_cleanup.assert_called_once_with(100)

    def test_error_handling_in_report_generation(
        self,
        report_generator: PerformanceReportGenerator,
        sample_benchmark_results: dict[str, Any],
    ) -> None:
        """Test error handling during report generation."""
        # Mock the existing data processor instance to raise an exception
        with patch.object(
            report_generator.data_processor, "process_benchmark_results"
        ) as mock_process_data:
            mock_process_data.side_effect = Exception("Processing error")

            with pytest.raises(Exception, match="Processing error"):
                report_generator.generate_comprehensive_report(
                    sample_benchmark_results
                )

    def test_component_initialization(
        self, report_generator: PerformanceReportGenerator
    ) -> None:
        """Test that all components are properly initialized."""
        assert hasattr(report_generator, "metrics_collector")
        assert hasattr(report_generator, "data_processor")
        assert hasattr(report_generator, "historical_manager")
        assert hasattr(report_generator, "analyzer")
        assert hasattr(report_generator, "visualizer")
        assert hasattr(report_generator, "formatter")
        assert hasattr(report_generator, "logger")
