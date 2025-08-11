"""Table generation utilities for experiment comparison."""

import logging
from typing import Any

import numpy as np


class TableUtils:
    """Table generation utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the table utilities."""
        self.logger = logging.getLogger(__name__)

        # Define metrics for table generation
        self.metrics = ["iou", "dice", "f1", "precision", "recall"]

    def calculate_table_statistics(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Calculate statistics for comparison table.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing table statistics
        """
        stats = {}

        for metric_name in self.metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if values:
                stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                }

        return stats

    def calculate_metric_correlations(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate correlations between different metrics.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary of metric correlation coefficients
        """
        correlations = {}

        for i, metric1 in enumerate(self.metrics):
            for metric2 in self.metrics[i + 1 :]:
                values1 = [
                    metrics.get(metric1, 0.0)
                    for metrics in metrics_data.values()
                ]
                values2 = [
                    metrics.get(metric2, 0.0)
                    for metrics in metrics_data.values()
                ]

                if len(values1) > 1 and len(values2) > 1:
                    try:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(correlation):
                            correlations[f"{metric1}_vs_{metric2}"] = (
                                correlation
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Correlation calculation failed for {metric1} vs {metric2}: {e}"
                        )

        return correlations

    def format_comparison_table(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Format data for comparison table display.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing formatted table data
        """
        table_data = {
            "headers": ["Experiment ID"] + self.metrics + ["Composite Score"],
            "rows": [],
            "summary_stats": self.calculate_table_statistics(metrics_data),
        }

        # Calculate composite scores for ranking
        composite_scores = {}
        for exp_id, metrics in metrics_data.items():
            total_score = 0.0
            for metric_name in self.metrics:
                value = metrics.get(metric_name, 0.0)
                # Simple normalization and weighting
                normalized_value = min(max(value, 0.0), 1.0)
                weight = 0.2  # Equal weight for all metrics
                total_score += normalized_value * weight

            composite_scores[exp_id] = total_score

        # Sort by composite score
        sorted_experiments = sorted(
            composite_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create table rows
        for exp_id, composite_score in sorted_experiments:
            metrics = metrics_data[exp_id]
            row = [exp_id]

            for metric_name in self.metrics:
                value = metrics.get(metric_name, 0.0)
                row.append(f"{value:.4f}")

            row.append(f"{composite_score:.4f}")
            table_data["rows"].append(row)

        return table_data

    def generate_table_summary(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Generate a summary of table data.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing table summary
        """
        summary = {
            "total_experiments": len(metrics_data),
            "metrics_analyzed": len(self.metrics),
            "best_performers": {},
            "worst_performers": {},
            "correlation_insights": {},
        }

        # Find best and worst performers for each metric
        for metric_name in self.metrics:
            values_with_ids = [
                (exp_id, metrics.get(metric_name, 0.0))
                for exp_id, metrics in metrics_data.items()
            ]

            if values_with_ids:
                # Best performer
                best_exp_id, best_value = max(
                    values_with_ids, key=lambda x: x[1]
                )
                summary["best_performers"][metric_name] = {
                    "experiment_id": best_exp_id,
                    "value": best_value,
                }

                # Worst performer
                worst_exp_id, worst_value = min(
                    values_with_ids, key=lambda x: x[1]
                )
                summary["worst_performers"][metric_name] = {
                    "experiment_id": worst_exp_id,
                    "value": worst_value,
                }

        # Calculate correlations
        correlations = self.calculate_metric_correlations(metrics_data)

        # Find strong correlations
        strong_correlations = {
            name: value
            for name, value in correlations.items()
            if abs(value) > 0.7
        }

        summary["correlation_insights"] = {
            "strong_correlations": strong_correlations,
            "total_correlations": len(correlations),
        }

        return summary

    def export_table_data(
        self,
        metrics_data: dict[str, dict[str, float]],
        format_type: str = "dict",
    ) -> Any:
        """Export table data in different formats.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries
            format_type: Export format ("dict", "list", "csv")

        Returns:
            Exported data in specified format
        """
        table_data = self.format_comparison_table(metrics_data)

        if format_type == "dict":
            return table_data
        elif format_type == "list":
            return [table_data["headers"]] + table_data["rows"]
        elif format_type == "csv":
            # Return CSV-like string
            csv_lines = [",".join(table_data["headers"])]
            for row in table_data["rows"]:
                csv_lines.append(",".join(str(cell) for cell in row))
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
