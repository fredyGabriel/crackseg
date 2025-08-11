"""Base classes and types for publication-ready figure generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from matplotlib.figure import Figure

from crackseg.reporting.config import ExperimentData, ReportConfig
from crackseg.reporting.utils.figures import (
    save_figure_multiple_formats,
    setup_publication_style,
)


@dataclass
class PublicationStyle:
    """Configuration for publication-ready figure styling."""

    # Figure dimensions
    figure_width: float = 6.0
    figure_height: float = 4.0
    dpi: int = 300

    # Typography
    font_family: str = "serif"
    font_size: int = 10
    title_font_size: int = 12
    legend_font_size: int = 9

    # Colors and styling
    color_palette: str = "viridis"
    line_width: float = 1.5
    marker_size: float = 6.0
    grid_alpha: float = 0.3

    # Export settings
    supported_formats: list[str] | None = None
    transparent_background: bool = True
    tight_layout: bool = True

    def __post_init__(self) -> None:
        if self.supported_formats is None:
            self.supported_formats = ["png", "svg", "pdf"]


class FigureDataProvider(Protocol):
    """Protocol for data providers that supply figure data."""

    def get_training_metrics(self, experiment_data: ExperimentData) -> Any: ...

    def get_performance_metrics(
        self, experiment_data: ExperimentData
    ) -> dict[str, float]: ...

    def get_comparison_data(
        self, experiments_data: list[ExperimentData]
    ) -> Any: ...


class BaseFigureGenerator:
    """Base class for publication-ready figure generators."""

    def __init__(self, style: PublicationStyle | None = None) -> None:
        self.style = style or PublicationStyle()
        self._setup_matplotlib_style()

    def _setup_matplotlib_style(self) -> None:
        setup_publication_style(self.style)

    def generate_figure(
        self,
        data: Any,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> dict[str, Path]:
        raise NotImplementedError

    def save_figure(
        self, fig: Figure, base_path: Path, formats: list[str] | None = None
    ) -> dict[str, Path]:
        return save_figure_multiple_formats(
            fig, base_path, formats, self.style
        )
