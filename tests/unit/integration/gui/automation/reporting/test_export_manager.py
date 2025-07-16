"""Unit tests for ExportManager component.

This module tests the export management system that handles multi-format
report exports (HTML, JSON, CSV).
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.integration.gui.automation.reporting.export_manager import (
    MultiFormatExportManager as ExportManager,
)


class TestExportManager:
    """Test suite for ExportManager functionality."""

    @pytest.fixture
    def export_manager(self) -> ExportManager:
        """Create ExportManager instance for testing."""
        return ExportManager()

    @pytest.fixture
    def sample_report_data(self) -> dict[str, Any]:
        """Provide sample report data for testing."""
        return {
            "executive_summary": {
                "overall_success_rate": 95.0,
                "critical_issues": 2,
                "recommendations": ["Optimize performance", "Fix memory leak"],
            },
            "technical_analysis": {
                "code_coverage": 85.0,
                "performance_metrics": {"avg_response_time": 120.5},
                "architecture_health": "good",
            },
            "operations_monitoring": {
                "resource_utilization": 70.0,
                "error_rates": {"critical": 0.1, "warning": 2.5},
                "deployment_status": "ready",
            },
            "metadata": {
                "generation_timestamp": "2025-01-07T02:00:00Z",
                "version": "1.0.0",
                "data_sources": ["automation", "monitoring", "testing"],
            },
        }

    def test_initialization(self, export_manager: ExportManager) -> None:
        """Test ExportManager initializes correctly."""
        assert export_manager.html_exporter is not None
        assert export_manager.json_exporter is not None
        assert export_manager.csv_exporter is not None

    def test_export_html_format(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test HTML export functionality."""
        result = export_manager.export_report(
            sample_report_data, format="html", output_dir=tmp_path
        )

        assert "path" in result
        assert "format" in result
        assert result["format"] == "html"

        output_path = Path(result["path"])
        assert output_path.exists()
        assert output_path.suffix == ".html"

        # Verify HTML content
        content = output_path.read_text(encoding="utf-8")
        assert "<html" in content
        assert "executive_summary" in content
        assert "95.0" in content  # Success rate

    def test_export_json_format(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test JSON export functionality."""
        result = export_manager.export_report(
            sample_report_data, format="json", output_dir=tmp_path
        )

        assert "path" in result
        assert "format" in result
        assert result["format"] == "json"

        output_path = Path(result["path"])
        assert output_path.exists()
        assert output_path.suffix == ".json"

        # Verify JSON content
        with open(output_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["executive_summary"]["overall_success_rate"] == 95.0
        assert loaded_data["technical_analysis"]["code_coverage"] == 85.0

    def test_export_csv_format(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test CSV export functionality."""
        result = export_manager.export_report(
            sample_report_data, format="csv", output_dir=tmp_path
        )

        assert "path" in result
        assert "format" in result
        assert result["format"] == "csv"

        output_path = Path(result["path"])
        assert output_path.exists()
        assert output_path.suffix == ".csv"

        # Verify CSV content
        content = output_path.read_text(encoding="utf-8")
        assert "section,metric,value" in content
        assert "executive_summary" in content
        assert "95.0" in content

    def test_export_multiple_formats(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test exporting to multiple formats."""
        formats = ["html", "json", "csv"]
        results = export_manager.export_multiple_formats(
            sample_report_data, formats=formats, output_dir=tmp_path
        )

        assert len(results) == 3
        assert all(format in results for format in formats)
        assert all("path" in results[fmt] for fmt in formats)
        assert all(Path(results[fmt]["path"]).exists() for fmt in formats)

    def test_export_with_custom_filename(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test export with custom filename."""
        custom_filename = "custom_report"
        result = export_manager.export_report(
            sample_report_data,
            format="json",
            output_dir=tmp_path,
            filename=custom_filename,
        )

        output_path = Path(result["path"])
        assert output_path.stem == custom_filename
        assert output_path.exists()

    def test_export_invalid_format(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_manager.export_report(
                sample_report_data, format="invalid", output_dir=tmp_path
            )

    def test_export_empty_data(
        self, export_manager: ExportManager, tmp_path: Path
    ) -> None:
        """Test export with empty data."""
        empty_data: dict[str, Any] = {}
        result = export_manager.export_report(
            empty_data, format="json", output_dir=tmp_path
        )

        output_path = Path(result["path"])
        assert output_path.exists()

        # Verify empty JSON
        with open(output_path, encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == {}

    def test_export_with_missing_output_dir(
        self, export_manager: ExportManager, sample_report_data: dict[str, Any]
    ) -> None:
        """Test export with missing output directory."""
        missing_dir = Path("non_existent_directory")
        result = export_manager.export_report(
            sample_report_data, format="json", output_dir=missing_dir
        )

        # Should create the directory and export successfully
        assert missing_dir.exists()
        assert Path(result["path"]).exists()

        # Cleanup
        import shutil

        shutil.rmtree(missing_dir)

    def test_get_supported_formats(
        self, export_manager: ExportManager
    ) -> None:
        """Test getting supported formats."""
        formats = export_manager.get_supported_formats()
        assert isinstance(formats, list)
        assert "html" in formats
        assert "json" in formats
        assert "csv" in formats

    def test_validate_export_data(
        self, export_manager: ExportManager, sample_report_data: dict[str, Any]
    ) -> None:
        """Test export data validation."""
        # Should not raise any exception
        export_manager._validate_export_data(sample_report_data)

    def test_validate_export_data_invalid(
        self, export_manager: ExportManager
    ) -> None:
        """Test export data validation with invalid data."""
        invalid_data = None
        with pytest.raises(ValueError, match="Export data cannot be None"):
            export_manager._validate_export_data(invalid_data)

    def test_generate_filename_with_timestamp(
        self, export_manager: ExportManager
    ) -> None:
        """Test filename generation with timestamp."""
        filename = export_manager._generate_filename("test", "json")
        assert filename.startswith("test_")
        assert filename.endswith(".json")
        assert len(filename) > len("test_.json")  # Should have timestamp

    def test_export_metadata_preservation(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that metadata is preserved in exports."""
        result = export_manager.export_report(
            sample_report_data, format="json", output_dir=tmp_path
        )

        with open(result["path"], encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert "metadata" in loaded_data
        assert loaded_data["metadata"]["version"] == "1.0.0"
        assert (
            loaded_data["metadata"]["generation_timestamp"]
            == "2025-01-07T02:00:00Z"
        )

    def test_export_large_data(
        self, export_manager: ExportManager, tmp_path: Path
    ) -> None:
        """Test export with large data set."""
        large_data = {
            "section": {f"metric_{i}": f"value_{i}" for i in range(1000)}
        }

        result = export_manager.export_report(
            large_data, format="json", output_dir=tmp_path
        )

        output_path = Path(result["path"])
        assert output_path.exists()

        # Verify large data is preserved
        with open(output_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert len(loaded_data["section"]) == 1000
        assert loaded_data["section"]["metric_500"] == "value_500"

    def test_export_error_handling(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test error handling during export."""
        # Mock file system error
        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                export_manager.export_report(
                    sample_report_data, format="json", output_dir=tmp_path
                )

    def test_export_concurrent_access(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test concurrent export operations."""
        # Simulate concurrent exports
        results: list[dict[str, Any]] = []
        for i in range(3):
            result = export_manager.export_report(
                sample_report_data,
                format="json",
                output_dir=tmp_path,
                filename=f"concurrent_test_{i}",
            )
            results.append(result)

        # All exports should succeed
        assert len(results) == 3
        assert all(Path(result["path"]).exists() for result in results)

    def test_export_cleanup_on_error(
        self,
        export_manager: ExportManager,
        sample_report_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test cleanup behavior on export errors."""
        # This test verifies that partial files are cleaned up on error
        with patch.object(
            export_manager.json_exporter, "export_json"
        ) as mock_export:
            mock_export.side_effect = Exception("Export failed")

            with pytest.raises(Exception, match="Export failed"):
                export_manager.export_report(
                    sample_report_data, format="json", output_dir=tmp_path
                )

            # Verify no partial files remain
            json_files = list(tmp_path.glob("*.json"))
            assert len(json_files) == 0
