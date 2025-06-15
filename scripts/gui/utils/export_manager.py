"""
This module defines the ExportManager for handling ZIP archive creation.
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scripts.gui.utils.results import ResultTriplet, TripletHealth

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Manages the creation of ZIP archives for experiment results.
    """

    def __init__(
        self,
        on_progress: Callable[[float, str], None] | None = None,
    ):
        """
        Initializes the ExportManager.

        Args:
            on_progress: Callback to report progress (percentage, message).
        """
        self.on_progress = on_progress
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create_export_zip(
        self,
        triplets: list[ResultTriplet],
        export_path: Path,
        include_images: bool = True,
        report_data: dict[str, Any] | None = None,
    ) -> Future[Path]:
        """
        Starts the ZIP export process in a background thread.

        Args:
            triplets: List of triplets to include in the export.
            export_path: The path to save the final ZIP file.
            include_images: Whether to include image files.
            report_data: Validated report data to include as report.json.

        Returns:
            A Future object that will contain the path to the created ZIP file.
        """
        future = self._executor.submit(
            self._zip_creation_task,
            triplets,
            export_path,
            include_images,
            report_data,
        )
        return future

    def _zip_creation_task(
        self,
        triplets: list[ResultTriplet],
        export_path: Path,
        include_images: bool,
        report_data: dict[str, Any] | None,
    ) -> Path:
        """
        The actual ZIP creation logic that runs in a background thread.
        """
        self._report_progress(0, "Starting export...")
        time.sleep(0.5)  # Give UI time to update

        # Gather files to zip, respecting health status
        files_to_process: list[tuple[Path, str]] = []
        for triplet in triplets:
            # Re-check health right before export
            triplet.check_health()
            if triplet.health_status == TripletHealth.BROKEN:
                logger.warning(
                    f"Skipping broken triplet {triplet.id} during export."
                )
                continue

            if triplet.health_status == TripletHealth.DEGRADED:
                logger.warning(
                    f"Triplet {triplet.id} is degraded; "
                    f"exporting only available files."
                )

            if include_images:
                potential_paths = [
                    triplet.image_path,
                    triplet.mask_path,
                    triplet.prediction_path,
                ]
                for path in potential_paths:
                    if path not in triplet.missing_files and path.exists():
                        # Create a unique path in the zip file
                        arcname = (
                            f"{triplet.dataset_name}/{triplet.id}/{path.name}"
                        )
                        files_to_process.append((path, arcname))

        # In a real scenario, we would also add config.yaml, metrics.json, etc.

        total_files = len(files_to_process)
        if total_files == 0 and not report_data:
            self._report_progress(1.0, "No valid files to export.")
            # Create an empty zip file to satisfy the future
            with zipfile.ZipFile(export_path, "w") as zipf:
                pass
            return export_path

        with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add report.json if data is provided
            if report_data:
                self._report_progress(0, "Generating report.json...")
                report_str = json.dumps(report_data, indent=4)
                zipf.writestr("report.json", report_str)

            for i, (file_path, arcname) in enumerate(files_to_process):
                zipf.write(file_path, arcname=arcname)
                message = f"Compressing {file_path.name}..."
                # Adjust progress calculation to account for report.json
                progress = (i + 1) / (total_files + 1)
                self._report_progress(progress, message)

        self._report_progress(1.0, "Export complete!")
        return export_path

    def _report_progress(self, percent: float, message: str) -> None:
        """Reports progress using the callback if available."""
        if self.on_progress:
            try:
                self.on_progress(percent, message)
            except Exception:
                logger.exception("Error in progress callback.")

    def shutdown(self) -> None:
        """Shuts down the thread pool executor."""
        self._executor.shutdown(wait=True)
