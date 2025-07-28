"""Core ExperimentReporter class for comprehensive experiment reporting.

This module provides the main ExperimentReporter class that orchestrates all
reporting components to generate comprehensive experiment reports.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    ExperimentData,
    OutputFormat,
    ReportConfig,
    ReportMetadata,
    TemplateType,
)
from .interfaces import (
    ComparisonEngine,
    DataLoader,
    FigureGenerator,
    PerformanceAnalyzer,
    RecommendationEngine,
    ReportExporter,
    SummaryGenerator,
    TemplateManager,
)


class ExperimentReporter:
    """
    Main orchestrator for comprehensive experiment reporting.

    This class coordinates all reporting components to generate executive
    summaries, detailed performance analysis, experiment comparisons, and
    publication-ready figures.
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        data_loader: DataLoader | None = None,
        summary_generator: SummaryGenerator | None = None,
        performance_analyzer: PerformanceAnalyzer | None = None,
        comparison_engine: ComparisonEngine | None = None,
        figure_generator: FigureGenerator | None = None,
        template_manager: TemplateManager | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        report_exporter: ReportExporter | None = None,
    ) -> None:
        """
        Initialize the ExperimentReporter.

        Args:
            config: Reporting configuration
            data_loader: Component for loading experiment data
            summary_generator: Component for generating summaries
            performance_analyzer: Component for performance analysis
            comparison_engine: Component for experiment comparison
            figure_generator: Component for generating figures
            template_manager: Component for template management
            recommendation_engine: Component for generating recommendations
            report_exporter: Component for exporting reports
        """
        self.config = config or ReportConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components (will be implemented in separate modules)
        self.data_loader = data_loader
        self.summary_generator = summary_generator
        self.performance_analyzer = performance_analyzer
        self.comparison_engine = comparison_engine
        self.figure_generator = figure_generator
        self.template_manager = template_manager
        self.recommendation_engine = recommendation_engine
        self.report_exporter = report_exporter

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "ExperimentReporter initialized with output_dir: "
            f"{self.config.output_dir}"
        )

    def generate_single_experiment_report(
        self,
        experiment_dir: Path,
        report_type: TemplateType | None = None,
    ) -> ReportMetadata:
        """
        Generate comprehensive report for a single experiment.

        Args:
            experiment_dir: Path to experiment directory
            report_type: Type of report to generate

        Returns:
            ReportMetadata with generation results
        """
        start_time = time.time()
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self.logger.info(
                f"Generating report for experiment: {experiment_dir}"
            )

            # Load experiment data
            experiment_data = self._load_experiment_data(experiment_dir)

            # Set report type
            report_type = report_type or self.config.template_type

            # Generate report content
            report_content = self._generate_report_content(
                [experiment_data], report_type
            )

            # Export reports
            output_files = self._export_reports(
                report_content, report_id, [experiment_data.experiment_id]
            )

            generation_time = time.time() - start_time

            return ReportMetadata(
                report_id=report_id,
                generation_timestamp=datetime.now().isoformat(),
                experiment_ids=[experiment_data.experiment_id],
                report_type=report_type,
                output_formats=self.config.output_formats,
                file_paths=output_files,
                generation_time_seconds=generation_time,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return ReportMetadata(
                report_id=report_id,
                generation_timestamp=datetime.now().isoformat(),
                experiment_ids=[experiment_dir.name],
                report_type=report_type or self.config.template_type,
                output_formats=self.config.output_formats,
                success=False,
                error_message=str(e),
            )

    def generate_comparison_report(
        self,
        experiment_dirs: list[Path],
        report_type: TemplateType | None = None,
    ) -> ReportMetadata:
        """
        Generate comparison report for multiple experiments.

        Args:
            experiment_dirs: List of experiment directories
            report_type: Type of report to generate

        Returns:
            ReportMetadata with generation results
        """
        start_time = time.time()
        report_id = (
            f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            self.logger.info(
                f"Generating comparison report for {len(experiment_dirs)} "
                "experiments"
            )

            # Load all experiment data
            experiments_data = self._load_multiple_experiments(experiment_dirs)

            # Set report type
            report_type = report_type or TemplateType.COMPARISON_REPORT

            # Generate report content
            report_content = self._generate_report_content(
                experiments_data, report_type
            )

            # Export reports
            experiment_ids = [exp.experiment_id for exp in experiments_data]
            output_files = self._export_reports(
                report_content, report_id, experiment_ids
            )

            generation_time = time.time() - start_time

            return ReportMetadata(
                report_id=report_id,
                generation_timestamp=datetime.now().isoformat(),
                experiment_ids=experiment_ids,
                report_type=report_type,
                output_formats=self.config.output_formats,
                file_paths=output_files,
                generation_time_seconds=generation_time,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {e}")
            return ReportMetadata(
                report_id=report_id,
                generation_timestamp=datetime.now().isoformat(),
                experiment_ids=[str(d) for d in experiment_dirs],
                report_type=report_type or TemplateType.COMPARISON_REPORT,
                output_formats=self.config.output_formats,
                success=False,
                error_message=str(e),
            )

    def _load_experiment_data(self, experiment_dir: Path) -> ExperimentData:
        """Load experiment data from directory."""
        if not self.data_loader:
            raise NotImplementedError("DataLoader component not implemented")

        return self.data_loader.load_experiment_data(experiment_dir)

    def _load_multiple_experiments(
        self, experiment_dirs: list[Path]
    ) -> list[ExperimentData]:
        """Load multiple experiments data."""
        if not self.data_loader:
            raise NotImplementedError("DataLoader component not implemented")

        return self.data_loader.load_multiple_experiments(experiment_dirs)

    def _generate_report_content(
        self,
        experiments_data: list[ExperimentData],
        report_type: TemplateType,
    ) -> dict[str, Any]:
        """Generate comprehensive report content."""
        content = {
            "report_type": report_type.value,
            "generation_timestamp": datetime.now().isoformat(),
            "experiments_count": len(experiments_data),
            "experiments": [exp.experiment_id for exp in experiments_data],
        }

        # Generate summaries
        if self.summary_generator:
            if len(experiments_data) == 1:
                content["summary"] = (
                    self.summary_generator.generate_executive_summary(
                        experiments_data[0], self.config
                    )
                )
            else:
                content["summary"] = (
                    self.summary_generator.generate_comparison_summary(
                        experiments_data, self.config
                    )
                )

        # Generate performance analysis
        if (
            self.performance_analyzer
            and self.config.include_performance_analysis
        ):
            content["performance_analysis"] = []
            for exp_data in experiments_data:
                analysis = self.performance_analyzer.analyze_performance(
                    exp_data, self.config
                )
                content["performance_analysis"].append(analysis)

        # Generate comparison data
        if self.comparison_engine and len(experiments_data) > 1:
            content["comparison"] = self.comparison_engine.compare_experiments(
                experiments_data, self.config
            )
            content["best_performing"] = (
                self.comparison_engine.identify_best_performing(
                    experiments_data, self.config
                )
            )

        # Generate recommendations
        if self.recommendation_engine and self.config.include_recommendations:
            content["recommendations"] = []
            for exp_data in experiments_data:
                recommendations = (
                    self.recommendation_engine.analyze_training_patterns(
                        exp_data, self.config
                    )
                )
                content["recommendations"].extend(recommendations)

        # Generate figures
        if self.config.include_publication_figures:
            content["figures"] = {}

            # Initialize publication figure generator if not provided
            if not self.figure_generator:
                from .figures import PublicationFigureGenerator

                self.figure_generator = PublicationFigureGenerator()

            # Generate figures for each experiment
            for exp_data in experiments_data:
                exp_figures = {}

                # Generate publication-ready figures
                publication_figures = (
                    self.figure_generator.generate_publication_figures(
                        exp_data, self.config
                    )
                )
                if publication_figures:
                    exp_figures["publication"] = publication_figures

                # Generate comparison figures if multiple experiments
                if len(experiments_data) > 1:
                    comparison_figures = (
                        self.figure_generator.generate_comparison_figures(
                            experiments_data, self.config
                        )
                    )
                    if comparison_figures:
                        exp_figures["comparison"] = comparison_figures

                content["figures"][exp_data.experiment_id] = exp_figures

        return content

    def _export_reports(
        self,
        content: dict[str, Any],
        report_id: str,
        experiment_ids: list[str],
    ) -> dict[OutputFormat, Path]:
        """Export reports in all configured formats."""
        if not self.report_exporter:
            raise NotImplementedError(
                "ReportExporter component not implemented"
            )

        output_files: dict[OutputFormat, Path] = {}

        for output_format in self.config.output_formats:
            try:
                output_path = (
                    self.config.output_dir
                    / f"{report_id}.{output_format.value}"
                )
                exported_path = self.report_exporter.export_report(
                    content, output_format.value, output_path, self.config
                )
                output_files[output_format] = exported_path

            except Exception as e:
                self.logger.error(
                    f"Failed to export {output_format.value} format: {e}"
                )

        return output_files

    def get_available_templates(self) -> list[str]:
        """Get list of available templates."""
        if not self.template_manager:
            return [template.value for template in TemplateType]

        return self.template_manager.get_available_templates()

    def validate_experiment_directory(self, experiment_dir: Path) -> bool:
        """
        Validate that an experiment directory contains required files.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            True if directory is valid, False otherwise
        """
        required_files = [
            "config.yaml",
            "metrics/complete_summary.json",
        ]

        for required_file in required_files:
            if not (experiment_dir / required_file).exists():
                self.logger.warning(f"Missing required file: {required_file}")
                return False

        return True
