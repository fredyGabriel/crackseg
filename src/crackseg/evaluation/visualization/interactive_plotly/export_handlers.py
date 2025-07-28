"""Export handlers for multi-format plot saving."""

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from matplotlib.figure import Figure


class ExportHandler:
    """Handle multi-format plot export with metadata preservation."""

    def __init__(self, export_formats: list[str]) -> None:
        """Initialize export handler.

        Args:
            export_formats: List of supported export formats.
        """
        self.export_formats = export_formats
        self.supported_formats = {"html", "png", "pdf", "svg", "jpg", "json"}

    def save_plot(
        self,
        fig: go.Figure | Figure,
        save_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save plot in multiple formats.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
            metadata: Optional metadata to embed.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in all requested formats
        for fmt in self.export_formats:
            if fmt not in self.supported_formats:
                continue

            if fmt == "html":
                self._save_html(fig, save_path, metadata)
            elif fmt == "png":
                self._save_png(fig, save_path)
            elif fmt == "pdf":
                self._save_pdf(fig, save_path)
            elif fmt == "svg":
                self._save_svg(fig, save_path)
            elif fmt == "jpg":
                self._save_jpg(fig, save_path)
            elif fmt == "json":
                self._save_json(fig, save_path, metadata)

    def _save_html(
        self,
        fig: go.Figure | Figure,
        save_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save plot as HTML with optional metadata.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
            metadata: Optional metadata to embed.
        """
        if hasattr(fig, "to_html"):
            # Plotly figure
            html_content = fig.to_html(include_plotlyjs=True, full_html=True)

            if metadata:
                html_content = self._embed_metadata_in_html(
                    html_content, metadata
                )

            with open(
                save_path.with_suffix(".html"), "w", encoding="utf-8"
            ) as f:
                f.write(html_content)

    def _save_png(self, fig: go.Figure | Figure, save_path: Path) -> None:
        """Save plot as PNG.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
        """
        if hasattr(fig, "write_image"):
            # Plotly figure
            fig.write_image(save_path.with_suffix(".png"))
        elif hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(
                save_path.with_suffix(".png"), dpi=300, bbox_inches="tight"
            )

    def _save_pdf(self, fig: go.Figure | Figure, save_path: Path) -> None:
        """Save plot as PDF.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
        """
        if hasattr(fig, "write_image"):
            # Plotly figure
            fig.write_image(save_path.with_suffix(".pdf"))
        elif hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(
                save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight"
            )

    def _save_svg(self, fig: go.Figure | Figure, save_path: Path) -> None:
        """Save plot as SVG.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
        """
        if hasattr(fig, "write_image"):
            # Plotly figure
            fig.write_image(save_path.with_suffix(".svg"))
        elif hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(
                save_path.with_suffix(".svg"), dpi=300, bbox_inches="tight"
            )

    def _save_jpg(self, fig: go.Figure | Figure, save_path: Path) -> None:
        """Save plot as JPG.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
        """
        if hasattr(fig, "write_image"):
            # Plotly figure
            fig.write_image(save_path.with_suffix(".jpg"))
        elif hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(
                save_path.with_suffix(".jpg"), dpi=300, bbox_inches="tight"
            )

    def _save_json(
        self,
        fig: go.Figure | Figure,
        save_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save plot data as JSON with metadata.

        Args:
            fig: Plotly figure or matplotlib figure.
            save_path: Base path for saving.
            metadata: Optional metadata to include.
        """
        if hasattr(fig, "to_json"):
            # Plotly figure
            plot_data = fig.to_json()
        elif isinstance(fig, Figure):
            # Matplotlib figure - convert to basic data
            plot_data = self._matplotlib_to_json(fig)
        else:
            # Fallback for other figure types
            plot_data = {
                "type": "unknown_figure",
                "error": "Unsupported figure type",
            }

        # Combine plot data with metadata
        export_data = {
            "plot_data": plot_data,
            "metadata": metadata or {},
            "export_info": {
                "format": "json",
                "timestamp": str(Path().cwd()),
            },
        }

        with open(save_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    def _embed_metadata_in_html(
        self, html_content: str, metadata: dict[str, Any]
    ) -> str:
        """Embed metadata in HTML content.

        Args:
            html_content: Original HTML content.
            metadata: Metadata to embed.

        Returns:
            HTML content with embedded metadata.
        """
        metadata_script = f"""
        <script>
        // Embedded metadata
        window.plotMetadata = {metadata};
        </script>
        """

        # Insert metadata before closing body tag
        if "</body>" in html_content:
            return html_content.replace("</body>", f"{metadata_script}</body>")
        else:
            return html_content + metadata_script

    def _matplotlib_to_json(self, fig: Figure) -> dict[str, Any]:
        """Convert matplotlib figure to JSON-serializable data.

        Args:
            fig: Matplotlib figure.

        Returns:
            JSON-serializable figure data.
        """
        # Basic conversion - in practice, you might want more sophisticated
        # conversion
        return {
            "type": "matplotlib_figure",
            "figure_size": fig.get_size_inches().tolist(),
            "axes_count": len(fig.axes),
            "dpi": fig.dpi,
        }
