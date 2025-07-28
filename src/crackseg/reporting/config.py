"""Configuration classes for the experimental reporting system."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class OutputFormat(Enum):
    """Supported output formats for reports."""

    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


class TemplateType(Enum):
    """Available template types for reports."""

    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    PUBLICATION_READY = "publication_ready"
    COMPARISON_REPORT = "comparison_report"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class ReportConfig:
    """Configuration for experiment reporting system."""

    # Output settings
    output_formats: list[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML]
    )
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    template_type: TemplateType = TemplateType.EXECUTIVE_SUMMARY

    # Content settings
    include_performance_analysis: bool = True
    include_comparison_charts: bool = True
    include_publication_figures: bool = True
    include_recommendations: bool = True
    include_trend_analysis: bool = True

    # Visualization settings
    figure_dpi: int = 300
    figure_format: str = "png"
    chart_theme: str = "plotly_white"
    color_palette: str = "viridis"

    # Analysis settings
    performance_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "iou_min": 0.7,
            "f1_min": 0.75,
            "precision_min": 0.8,
            "recall_min": 0.7,
        }
    )
    trend_analysis_window: int = 10
    anomaly_detection_enabled: bool = True

    # Template settings
    custom_templates_dir: Path | None = None
    default_template_vars: dict[str, Any] = field(default_factory=dict)

    # Export settings
    compress_outputs: bool = True
    include_metadata: bool = True
    auto_cleanup_old_reports: bool = True
    retention_days: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if (
            self.custom_templates_dir
            and not self.custom_templates_dir.exists()
        ):
            raise ValueError(
                "Custom templates directory does not exist: "
                f"{self.custom_templates_dir}"
            )


@dataclass
class ExperimentData:
    """Data structure for experiment information."""

    experiment_id: str
    experiment_dir: Path
    config: DictConfig
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate experiment data."""
        if not self.experiment_dir.exists():
            raise ValueError(
                f"Experiment directory does not exist: {self.experiment_dir}"
            )

        # Metrics can be empty for incomplete experiments
        pass


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""

    report_id: str
    generation_timestamp: str
    experiment_ids: list[str]
    report_type: TemplateType
    output_formats: list[OutputFormat]
    file_paths: dict[OutputFormat, Path] = field(default_factory=dict)
    generation_time_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
