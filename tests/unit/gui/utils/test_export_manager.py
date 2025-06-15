"""Unit tests for the ExportManager class."""

import json
import zipfile
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from scripts.gui.utils.export_manager import ExportManager


class MockTripletHealth(Enum):
    """Mock enum for TripletHealth."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BROKEN = "broken"


class MockResultTriplet:
    """Mock class for ResultTriplet."""

    def __init__(
        self,
        triplet_id: str,
        dataset: str,
        base_path: Path,
        health: MockTripletHealth,
        missing: list[str] | None = None,
    ):
        self.id = triplet_id
        self.dataset_name = dataset
        self.health_status = health
        self.image_path = base_path / f"{triplet_id}_image.png"
        self.mask_path = base_path / f"{triplet_id}_mask.png"
        self.prediction_path = base_path / f"{triplet_id}_pred.png"
        self.missing_files: list[Path] = []
        if missing:
            for file_type in missing:
                if file_type == "image":
                    self.missing_files.append(self.image_path)
                elif file_type == "mask":
                    self.missing_files.append(self.mask_path)
                elif file_type == "prediction":
                    self.missing_files.append(self.prediction_path)

        # Create the files that are not missing
        for path in [self.image_path, self.mask_path, self.prediction_path]:
            if path not in self.missing_files:
                path.touch()

    def check_health(self):
        """Mock check_health method."""
        pass


@pytest.fixture
def mock_triplets(tmp_path: Path) -> list[MockResultTriplet]:
    """Fixture to create mock triplets for testing."""
    dataset_path = tmp_path / "dataset1"
    dataset_path.mkdir()
    return [
        MockResultTriplet(
            "t1", "dataset1", dataset_path, MockTripletHealth.HEALTHY
        ),
        MockResultTriplet(
            "t2",
            "dataset1",
            dataset_path,
            MockTripletHealth.DEGRADED,
            missing=["mask"],
        ),
        MockResultTriplet(
            "t3", "dataset1", dataset_path, MockTripletHealth.BROKEN
        ),
    ]


@patch("scripts.gui.utils.export_manager.ResultTriplet", new=MockResultTriplet)
@patch("scripts.gui.utils.export_manager.TripletHealth", new=MockTripletHealth)
def test_zip_creation_task_full_export(
    tmp_path: Path, mock_triplets: list[MockResultTriplet]
):
    """Test the zip creation task with default options."""
    export_path = tmp_path / "export.zip"
    progress_callback = MagicMock()
    manager = ExportManager(on_progress=progress_callback)

    # We test the internal task method directly to avoid thread complexity
    manager._zip_creation_task(mock_triplets, export_path, True, None)  # type: ignore

    assert export_path.exists()
    with zipfile.ZipFile(export_path, "r") as zf:
        # 3 files for healthy, 2 for degraded, 0 for broken
        assert len(zf.namelist()) == 5
        assert "dataset1/t1/t1_image.png" in zf.namelist()
        assert "dataset1/t1/t1_mask.png" in zf.namelist()
        assert "dataset1/t1/t1_pred.png" in zf.namelist()
        assert "dataset1/t2/t2_image.png" in zf.namelist()
        assert "dataset1/t2/t2_pred.png" in zf.namelist()

    # Check progress calls
    assert progress_callback.call_count > 2
    # Check first and last calls specifically
    assert progress_callback.call_args_list[0] == call(0, "Starting export...")
    assert progress_callback.call_args_list[-1] == call(
        1.0, "Export complete!"
    )


@patch("scripts.gui.utils.export_manager.TripletHealth", new=MockTripletHealth)
def test_zip_creation_with_report(
    tmp_path: Path, mock_triplets: list[MockResultTriplet]
):
    """Test that report.json is correctly added to the zip."""
    export_path = tmp_path / "export_with_report.zip"
    report_data = {"test": "data", "value": 123}
    manager = ExportManager()

    manager._zip_creation_task(mock_triplets, export_path, False, report_data)  # type: ignore

    assert export_path.exists()
    with zipfile.ZipFile(export_path, "r") as zf:
        assert "report.json" in zf.namelist()
        with zf.open("report.json") as f:
            content = json.load(f)
            assert content == report_data
        # No images included
        assert len(zf.namelist()) == 1


@patch("scripts.gui.utils.export_manager.TripletHealth", new=MockTripletHealth)
def test_zip_creation_no_valid_files(tmp_path: Path):
    """Test zip creation when no valid files are available."""
    export_path = tmp_path / "empty.zip"
    manager = ExportManager()
    broken_triplet = MockResultTriplet(
        "t1", "d1", tmp_path, MockTripletHealth.BROKEN
    )

    manager._zip_creation_task([broken_triplet], export_path, True, None)  # type: ignore

    assert export_path.exists()
    with zipfile.ZipFile(export_path, "r") as zf:
        assert len(zf.namelist()) == 0
