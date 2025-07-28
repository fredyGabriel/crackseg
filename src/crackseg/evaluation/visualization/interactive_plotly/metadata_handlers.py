"""Metadata handlers for visualization components."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


class MetadataHandler:
    """Handle metadata for visualization components."""

    def __init__(self) -> None:
        """Initialize metadata handler."""
        self.default_metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "tool": "CrackSeg Visualization",
        }

    def create_metadata(
        self,
        plot_type: str,
        data_info: dict[str, Any] | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create metadata for a plot.

        Args:
            plot_type: Type of plot (e.g., 'training_curves',
                'prediction_grid').
            data_info: Information about the data used in the plot.
            custom_metadata: Additional custom metadata.

        Returns:
            Combined metadata dictionary.
        """
        metadata = self.default_metadata.copy()
        metadata["plot_type"] = plot_type
        metadata["data_info"] = data_info if data_info is not None else {}

        if custom_metadata:
            metadata.update(custom_metadata)

        return metadata

    def embed_metadata_in_figure(
        self,
        fig: go.Figure,
        metadata: dict[str, Any],
    ) -> go.Figure:
        """Embed metadata in a Plotly figure.

        Args:
            fig: Plotly figure.
            metadata: Metadata to embed.

        Returns:
            Figure with embedded metadata.
        """
        # Add metadata as annotation (hidden)
        fig.add_annotation(
            text="",
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            showarrow=False,
            visible=False,
            customdata=metadata,
        )

        return fig

    def extract_metadata_from_figure(self, fig: go.Figure) -> dict[str, Any]:
        """Extract metadata from a Plotly figure.

        Args:
            fig: Plotly figure.

        Returns:
            Extracted metadata dictionary.
        """
        metadata = {}

        # Try to extract from annotations
        for annotation in fig.layout.annotations:
            if hasattr(annotation, "customdata") and annotation.customdata:
                metadata.update(annotation.customdata)

        return metadata

    def save_metadata(
        self,
        metadata: dict[str, Any],
        save_path: Path,
    ) -> None:
        """Save metadata to a JSON file.

        Args:
            metadata: Metadata to save.
            save_path: Path to save the metadata file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        """Load metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata file.

        Returns:
            Loaded metadata dictionary.
        """
        metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            return {}

        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    def validate_metadata(self, metadata: dict[str, Any]) -> bool:
        """Validate metadata structure.

        Args:
            metadata: Metadata to validate.

        Returns:
            True if metadata is valid, False otherwise.
        """
        required_keys = {"plot_type", "created_at"}

        if not all(key in metadata for key in required_keys):
            return False

        if not isinstance(metadata.get("plot_type"), str):
            return False

        return True

    def merge_metadata(
        self,
        base_metadata: dict[str, Any],
        additional_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge two metadata dictionaries.

        Args:
            base_metadata: Base metadata dictionary.
            additional_metadata: Additional metadata to merge.

        Returns:
            Merged metadata dictionary.
        """
        merged = base_metadata.copy()

        for key, value in additional_metadata.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self.merge_metadata(merged[key], value)
            else:
                merged[key] = value

        return merged
