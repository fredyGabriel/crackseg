"""
Export service for the Results Gallery.

This service handles the logic for preparing data and generating
export files (e.g., ZIP, CSV) in a background thread to avoid
blocking the UI.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import scripts.gui.components.gallery.state_manager as sm
    from scripts.gui.utils.export_manager import ExportManager
    from scripts.gui.utils.results import ResultTriplet, TripletHealth


class GalleryExportService:
    """Manages the result export process."""

    def __init__(self, state_manager: sm.GalleryStateManager) -> None:
        """Initialize the export service."""
        self.state = state_manager

    def handle_export(
        self,
        scope: str,
        include_images: bool,
        include_metadata: bool,
    ) -> None:
        """Handle the export process in a background thread."""
        all_triplets = self.state.get("scan_results", [])
        active_filters = self.state.get("active_filters", {})

        if scope == "Selected Items":
            selected_ids = self.state.get("selected_triplet_ids")
            target_triplets = [t for t in all_triplets if t.id in selected_ids]
        elif scope == "Filtered Items":
            target_triplets = self._apply_filters(all_triplets, active_filters)
        else:  # All Items
            target_triplets = all_triplets

        if not target_triplets:
            # Handle no selection case, maybe with a toast
            return

        report_dict = self._create_report_dict(
            target_triplets, scope, include_images, include_metadata
        )

        export_manager: ExportManager = self.state.get("export_manager")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_path = Path(f"exports/crackseg_export_{timestamp}.zip")
        export_path.parent.mkdir(exist_ok=True)

        future = export_manager.create_export_zip(
            triplets=target_triplets,
            export_path=export_path,
            include_images=include_images,
            report_data=report_dict,
        )
        self.state.set("export_future", future)

    def _create_report_dict(
        self,
        triplets: list[ResultTriplet],
        scope: str,
        include_images: bool,
        include_metadata: bool,
    ) -> dict[str, Any]:
        """Create the data dictionary for the report model."""
        report_triplets_data = []
        for t in triplets:
            report_triplets_data.append(
                {
                    "id": t.id,
                    "dataset_name": t.dataset_name,
                    "health_status": t.health_status.value,
                    "is_complete": t.health_status == TripletHealth.HEALTHY,
                    "paths": {
                        "image": str(t.image_path),
                        "mask": str(t.mask_path),
                        "prediction": str(t.prediction_path),
                    },
                    "metadata": dict(t.metadata) if include_metadata else {},
                }
            )

        validation_stats = self.state.get("validation_stats", {})
        summary_data = {
            "total_triplets": validation_stats.get("total_triplets", 0),
            "valid_triplets": validation_stats.get("valid_triplets", 0),
            "export_scope": scope,
            "included_images": include_images,
            "included_metadata": include_metadata,
        }

        report_dict = {
            "summary": summary_data,
            "triplets": report_triplets_data,
        }

        # This part should be removed, validation happens in the model
        # validated_data, _ = create_report_data_model(report_dict)
        return report_dict

    def _apply_filters(
        self, triplets: list[ResultTriplet], active_filters: dict[str, Any]
    ) -> list[ResultTriplet]:
        """Apply active filters to the list of triplets."""
        if not active_filters:
            return triplets

        filtered_list = triplets
        if "search" in active_filters:
            term = active_filters["search"].lower()
            filtered_list = [t for t in filtered_list if term in t.id.lower()]
        if "health" in active_filters:
            health_statuses = set(active_filters["health"])
            filtered_list = [
                t
                for t in filtered_list
                if t.health_status.name in health_statuses
            ]
        return filtered_list
