"""Interface definitions for the experimental reporting system.

This module defines the core interfaces that all reporting components must
implement, ensuring consistency and extensibility across the reporting system.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from .config import ExperimentData, ReportConfig


class SummaryGenerator(ABC):
    """Interface for generating executive summaries."""

    @abstractmethod
    def generate_executive_summary(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Generate executive summary for a single experiment."""
        pass

    @abstractmethod
    def generate_comparison_summary(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Generate summary comparing multiple experiments."""
        pass


class PerformanceAnalyzer(ABC):
    """Interface for performance analysis and metrics evaluation."""

    @abstractmethod
    def analyze_performance(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Analyze performance metrics and generate insights."""
        pass

    @abstractmethod
    def detect_anomalies(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Detect performance anomalies across experiments."""
        pass

    @abstractmethod
    def generate_recommendations(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        pass


class ComparisonEngine(ABC):
    """Interface for experiment comparison and analysis."""

    @abstractmethod
    def compare_experiments(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Compare multiple experiments and generate analysis."""
        pass

    @abstractmethod
    def identify_best_performing(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Identify the best performing experiment based on criteria."""
        pass

    @abstractmethod
    def generate_comparison_table(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Generate tabular comparison of experiments."""
        pass


class FigureGenerator(ABC):
    """Interface for generating publication-ready figures."""

    @abstractmethod
    def generate_training_curves(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> Path:
        """Generate training curves figure."""
        pass

    @abstractmethod
    def generate_performance_charts(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> Path:
        """Generate performance comparison charts."""
        pass

    @abstractmethod
    def generate_publication_figures(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> dict[str, Path]:
        """Generate publication-ready figures."""
        pass


class TemplateManager(ABC):
    """Interface for template management and rendering."""

    @abstractmethod
    def load_template(
        self,
        template_type: str,
        config: ReportConfig,
    ) -> str:
        """Load template content."""
        pass

    @abstractmethod
    def render_template(
        self,
        template_content: str,
        data: dict[str, Any],
        config: ReportConfig,
    ) -> str:
        """Render template with data."""
        pass

    @abstractmethod
    def get_available_templates(self) -> list[str]:
        """Get list of available templates."""
        pass


class RecommendationEngine(ABC):
    """Interface for generating automated recommendations."""

    @abstractmethod
    def analyze_training_patterns(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> list[str]:
        """Analyze training patterns and generate recommendations."""
        pass

    @abstractmethod
    def suggest_hyperparameter_improvements(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> dict[str, Any]:
        """Suggest hyperparameter improvements."""
        pass

    @abstractmethod
    def identify_optimization_opportunities(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> list[str]:
        """Identify optimization opportunities."""
        pass


class DataLoader(Protocol):
    """Protocol for data loading components."""

    def load_experiment_data(self, experiment_dir: Path) -> ExperimentData:
        """Load experiment data from directory."""
        ...

    def load_multiple_experiments(
        self,
        experiment_dirs: list[Path],
    ) -> list[ExperimentData]:
        """Load multiple experiments data."""
        ...


class ReportExporter(Protocol):
    """Protocol for report export components."""

    def export_report(
        self,
        content: dict[str, Any],
        output_format: str,
        output_path: Path,
        config: ReportConfig,
    ) -> Path:
        """Export report in specified format."""
        ...

    def export_multiple_formats(
        self,
        content: dict[str, Any],
        output_formats: list[str],
        output_dir: Path,
        config: ReportConfig,
    ) -> dict[str, Path]:
        """Export report in multiple formats."""
        ...
