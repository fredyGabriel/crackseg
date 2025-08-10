"""
Artifact management integration for comparison system.

This module integrates the existing ArtifactManager with the comparison
system to provide comprehensive artifact management capabilities for
experiment comparison and reporting.

This implements TAREA 10.1: Sistema de gestión de artefactos
"""

import logging
from datetime import datetime
from typing import Any

from crackseg.utils.artifact_manager import (
    ArtifactManager,
    ArtifactManagerConfig,
)
from crackseg.utils.artifact_manager.metadata import ArtifactMetadata


class ComparisonArtifactManager:
    """
    Artifact management system integrated with comparison engine.

    This class provides seamless integration between the existing ArtifactManager
    and the comparison system, enabling:
    - Automatic artifact tracking during comparisons
    - Metadata enrichment for comparison artifacts
    - Version control for comparison results
    - Integration with experiment tracking
    """

    def __init__(
        self,
        base_path: str = "artifacts/comparisons",
        experiment_name: str | None = None,
        auto_create_dirs: bool = True,
        validate_on_save: bool = True,
        enable_versioning: bool = True,
    ) -> None:
        """
        Initialize the comparison artifact manager.

        Args:
            base_path: Base path for comparison artifacts
            experiment_name: Name for the comparison experiment
            auto_create_dirs: Whether to auto-create directories
            validate_on_save: Whether to validate artifacts on save
            enable_versioning: Whether to enable artifact versioning
        """
        self.logger = logging.getLogger(__name__)

        # Create configuration for comparison artifacts
        config = ArtifactManagerConfig(
            base_path=base_path,
            experiment_name=experiment_name
            or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            auto_create_dirs=auto_create_dirs,
            validate_on_save=validate_on_save,
            enable_versioning=enable_versioning,
        )

        # Initialize the core artifact manager
        self.artifact_manager = ArtifactManager(config)

        # Create comparison-specific directories
        self._ensure_comparison_directories()

        self.logger.info(
            f"ComparisonArtifactManager initialized: {self.artifact_manager.experiment_path}"
        )

    def _ensure_comparison_directories(self) -> None:
        """Create comparison-specific directory structure."""
        comparison_dirs = [
            "comparison_reports",
            "comparison_tables",
            "comparison_charts",
            "comparison_metrics",
            "comparison_artifacts",
            "comparison_logs",
        ]

        for directory in comparison_dirs:
            dir_path = self.artifact_manager.experiment_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_comparison_report(
        self,
        report_data: dict[str, Any],
        report_name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a comparison report as an artifact.

        Args:
            report_data: The comparison report data
            report_name: Name for the report
            description: Description of the report
            tags: Optional tags for categorization

        Returns:
            Path to the saved report artifact
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.json"

        # Save report to comparison_reports directory
        report_path = (
            self.artifact_manager.experiment_path
            / "comparison_reports"
            / filename
        )

        # Use artifact manager to save with metadata
        artifact_id = self.artifact_manager.storage.save_json(
            data=report_data,
            file_path=report_path,
            description=description or f"Comparison report: {report_name}",
            artifact_type="comparison_report",
            tags=tags or ["comparison", "report"],
        )

        self.logger.info(f"Saved comparison report: {artifact_id}")
        return str(report_path)

    def save_comparison_table(
        self,
        table_data: dict[str, Any],
        table_name: str,
        format_type: str = "csv",
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a comparison table as an artifact.

        Args:
            table_data: The comparison table data
            table_name: Name for the table
            format_type: Output format (csv, excel, json)
            description: Description of the table
            tags: Optional tags for categorization

        Returns:
            Path to the saved table artifact
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_{timestamp}.{format_type}"

        # Save table to comparison_tables directory
        table_path = (
            self.artifact_manager.experiment_path
            / "comparison_tables"
            / filename
        )

        # Use artifact manager to save with metadata
        artifact_id = self.artifact_manager.storage.save_data(
            data=table_data,
            file_path=table_path,
            description=description or f"Comparison table: {table_name}",
            artifact_type="comparison_table",
            tags=tags or ["comparison", "table"],
        )

        self.logger.info(f"Saved comparison table: {artifact_id}")
        return str(table_path)

    def save_comparison_chart(
        self,
        chart_data: bytes,
        chart_name: str,
        chart_format: str = "png",
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a comparison chart as an artifact.

        Args:
            chart_data: The chart data as bytes
            chart_name: Name for the chart
            chart_format: Chart format (png, jpg, svg, pdf)
            description: Description of the chart
            tags: Optional tags for categorization

        Returns:
            Path to the saved chart artifact
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_name}_{timestamp}.{chart_format}"

        # Save chart to comparison_charts directory
        chart_path = (
            self.artifact_manager.experiment_path
            / "comparison_charts"
            / filename
        )

        # Use artifact manager to save with metadata
        artifact_id = self.artifact_manager.storage.save_binary(
            data=chart_data,
            file_path=chart_path,
            description=description or f"Comparison chart: {chart_name}",
            artifact_type="comparison_chart",
            tags=tags or ["comparison", "chart"],
        )

        self.logger.info(f"Saved comparison chart: {artifact_id}")
        return str(chart_path)

    def save_comparison_metrics(
        self,
        metrics_data: dict[str, Any],
        metrics_name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Save comparison metrics as an artifact.

        Args:
            metrics_data: The comparison metrics data
            metrics_name: Name for the metrics
            description: Description of the metrics
            tags: Optional tags for categorization

        Returns:
            Path to the saved metrics artifact
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metrics_name}_{timestamp}.json"

        # Save metrics to comparison_metrics directory
        metrics_path = (
            self.artifact_manager.experiment_path
            / "comparison_metrics"
            / filename
        )

        # Use artifact manager to save with metadata
        artifact_id = self.artifact_manager.storage.save_json(
            data=metrics_data,
            file_path=metrics_path,
            description=description or f"Comparison metrics: {metrics_name}",
            artifact_type="comparison_metrics",
            tags=tags or ["comparison", "metrics"],
        )

        self.logger.info(f"Saved comparison metrics: {artifact_id}")
        return str(metrics_path)

    def get_comparison_artifacts(
        self,
        artifact_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ArtifactMetadata]:
        """
        Get comparison artifacts with optional filtering.

        Args:
            artifact_type: Filter by artifact type
            tags: Filter by tags

        Returns:
            List of matching artifact metadata
        """
        artifacts = self.artifact_manager.list_artifacts(artifact_type)

        # Filter by tags if specified
        if tags:
            artifacts = [
                artifact
                for artifact in artifacts
                if any(tag in artifact.tags for tag in tags)
            ]

        return artifacts

    def get_comparison_summary(self) -> dict[str, Any]:
        """
        Get a summary of all comparison artifacts.

        Returns:
            Dictionary containing artifact summary information
        """
        all_artifacts = self.artifact_manager.list_artifacts()

        # Group by artifact type
        type_counts = {}
        total_size = 0

        for artifact in all_artifacts:
            artifact_type = artifact.artifact_type
            type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
            total_size += artifact.file_size

        return {
            "total_artifacts": len(all_artifacts),
            "total_size_bytes": total_size,
            "type_distribution": type_counts,
            "experiment_path": str(self.artifact_manager.experiment_path),
            "created_at": self.artifact_manager.experiment_name,
            "last_updated": datetime.now().isoformat(),
        }

    def cleanup_old_artifacts(
        self,
        days_to_keep: int = 30,
        artifact_types: list[str] | None = None,
    ) -> int:
        """
        Clean up old comparison artifacts.

        Args:
            days_to_keep: Number of days to keep artifacts
            artifact_types: Specific artifact types to clean up

        Returns:
            Number of artifacts cleaned up
        """
        # This would implement cleanup logic
        # For now, return 0 as placeholder
        self.logger.info(
            f"Cleanup requested for artifacts older than {days_to_keep} days"
        )
        return 0

    def export_comparison_artifacts(
        self,
        export_path: str,
        artifact_types: list[str] | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Export comparison artifacts to a specified location.

        Args:
            export_path: Path to export artifacts to
            artifact_types: Specific artifact types to export
            include_metadata: Whether to include metadata files

        Returns:
            Path to the exported artifacts
        """
        # This would implement export logic
        # For now, return the export path as placeholder
        self.logger.info(f"Export requested to: {export_path}")
        return export_path
