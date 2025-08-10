"""Automated experiment comparison engine for crack segmentation.

This module provides a comprehensive comparison engine that automatically
analyzes multiple experiments, identifies best performers, and generates
detailed comparison reports with statistical analysis.

This refactored version uses the existing analysis modules to avoid
code duplication and maintain modularity.
"""

import logging
from typing import Any

import pandas as pd

from ..config import ExperimentData, ReportConfig
from ..interfaces import ComparisonEngine
from .analysis.anomalies import AnomalyDetector
from .analysis.ranking import RankingAnalyzer
from .analysis.statistical import StatisticalAnalyzer
from .analysis.trends import TrendAnalyzer


class AutomatedComparisonEngine(ComparisonEngine):
    """
    Automated experiment comparison engine with advanced analysis capabilities.

    This engine provides:
    - Statistical comparison of experiments
    - Multi-criteria ranking algorithms
    - Automated best performer identification
    - Detailed comparison tables and reports
    - Performance trend analysis
    """

    def __init__(self) -> None:
        """Initialize the automated comparison engine."""
        self.logger = logging.getLogger(__name__)

        # Initialize analysis modules
        self.anomaly_detector = AnomalyDetector()
        self.ranking_analyzer = RankingAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.trend_analyzer = TrendAnalyzer()

    def compare_experiments(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Compare multiple experiments and generate comprehensive analysis.

        Args:
            experiments_data: List of experiment data to compare
            config: Reporting configuration

        Returns:
            Dictionary containing comprehensive comparison analysis
        """
        self.logger.info(f"Comparing {len(experiments_data)} experiments")

        if len(experiments_data) < 2:
            return {"error": "Need at least 2 experiments for comparison"}

        comparison_results = {
            "experiment_count": len(experiments_data),
            "comparison_timestamp": pd.Timestamp.now().isoformat(),
            "statistical_analysis": {},
            "ranking_analysis": {},
            "performance_trends": {},
            "anomaly_detection": {},
            "recommendations": [],
        }

        # Extract metrics for comparison
        metrics_data = self._extract_comparison_metrics(experiments_data)

        # Perform statistical analysis using existing module
        comparison_results["statistical_analysis"] = (
            self.statistical_analyzer.perform_statistical_analysis(
                metrics_data
            )
        )

        # Generate ranking analysis using existing module
        comparison_results["ranking_analysis"] = (
            self.ranking_analyzer.generate_ranking_analysis(metrics_data)
        )

        # Analyze performance trends using existing module
        comparison_results["performance_trends"] = (
            self.trend_analyzer.analyze_performance_trends(experiments_data)
        )

        # Detect anomalies using existing module
        comparison_results["anomaly_detection"] = (
            self.anomaly_detector.detect_anomalies(metrics_data)
        )

        # Generate recommendations
        comparison_results["recommendations"] = (
            self._generate_comparison_recommendations(
                metrics_data, comparison_results
            )
        )

        return comparison_results

    def identify_best_performing(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Identify the best performing experiment based on comprehensive analysis.

        Args:
            experiments_data: List of experiment data to analyze
            config: Reporting configuration

        Returns:
            Dictionary containing best performer analysis
        """
        if len(experiments_data) < 1:
            return {"error": "No experiments to analyze"}

        # Extract metrics for comparison
        metrics_data = self._extract_comparison_metrics(experiments_data)

        # Use existing ranking analyzer to identify best performer
        best_performer = self.ranking_analyzer.identify_best_performer(
            metrics_data
        )

        # Add statistical significance check
        if "error" not in best_performer:
            composite_scores = (
                self.ranking_analyzer._calculate_composite_scores(metrics_data)
            )
            best_performer["statistically_significant"] = (
                self.statistical_analyzer.check_statistical_significance(
                    best_performer["best_experiment_id"], composite_scores
                )
            )
            best_performer["confidence_level"] = (
                self.statistical_analyzer.calculate_confidence_level(
                    best_performer["best_experiment_id"], composite_scores
                )
            )

        return best_performer

    def generate_comparison_table(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive comparison table for experiments.

        Args:
            experiments_data: List of experiment data to compare
            config: Reporting configuration

        Returns:
            Dictionary containing comparison table data
        """
        if len(experiments_data) < 2:
            return {"error": "Need at least 2 experiments for comparison"}

        # Extract metrics for comparison
        metrics_data = self._extract_comparison_metrics(experiments_data)

        # Generate table data
        table_data = {
            "headers": ["Experiment ID"]
            + list(next(iter(metrics_data.values())).keys()),
            "rows": [],
            "statistics": self._calculate_table_statistics(metrics_data),
        }

        # Add experiment rows
        for exp_id, metrics in metrics_data.items():
            row = [exp_id] + [
                metrics.get(metric, 0.0)
                for metric in table_data["headers"][1:]
            ]
            table_data["rows"].append(row)

        return table_data

    def _extract_comparison_metrics(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, dict[str, float]]:
        """
        Extract comparison metrics from experiment data.

        Args:
            experiments_data: List of experiment data

        Returns:
            Dictionary mapping experiment IDs to metric dictionaries
        """
        metrics_data = {}

        for exp_data in experiments_data:
            exp_id = exp_data.experiment_id
            metrics_data[exp_id] = {}

            if "complete_summary" in exp_data.metrics:
                summary = exp_data.metrics["complete_summary"]
                best_metrics = summary.get("best_metrics", {})

                for metric_name, metric_data in best_metrics.items():
                    if (
                        isinstance(metric_data, dict)
                        and "value" in metric_data
                    ):
                        metrics_data[exp_id][metric_name] = metric_data[
                            "value"
                        ]
                    elif isinstance(metric_data, int | float):
                        metrics_data[exp_id][metric_name] = float(metric_data)

        return metrics_data

    def _generate_comparison_recommendations(
        self,
        metrics_data: dict[str, dict[str, float]],
        comparison_results: dict[str, Any],
    ) -> list[str]:
        """
        Generate recommendations based on comparison analysis.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries
            comparison_results: Results from comparison analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Add recommendations based on anomaly detection
        anomalies = comparison_results.get("anomaly_detection", {})
        if anomalies.get("suspicious_experiments"):
            recommendations.append(
                "Review suspicious experiments for potential data quality issues"
            )

        if anomalies.get("performance_gaps"):
            recommendations.append(
                "Investigate significant performance gaps between experiments"
            )

        # Add recommendations based on trends
        trends = comparison_results.get("performance_trends", {})
        if trends.get("plateau_detected"):
            recommendations.append(
                "Performance plateau detected - consider exploring new architectures"
            )

        # Add recommendations based on statistical significance
        ranking = comparison_results.get("ranking_analysis", {})
        if ranking.get("ranking"):
            best_exp = ranking["ranking"][0]
            if best_exp.get("score_breakdown", {}).get("total_score", 0) < 0.8:
                recommendations.append(
                    "Best performer score is below 0.8 - room for improvement"
                )

        return recommendations

    def _calculate_table_statistics(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """
        Calculate statistics for comparison table.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing table statistics
        """
        if not metrics_data:
            return {}

        # Get all metric names
        all_metrics = set()
        for metrics in metrics_data.values():
            all_metrics.update(metrics.keys())

        statistics = {}
        for metric_name in all_metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]
            if values:
                statistics[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (
                        (
                            sum(
                                (x - sum(values) / len(values)) ** 2
                                for x in values
                            )
                            / len(values)
                        )
                        ** 0.5
                        if len(values) > 1
                        else 0.0
                    ),
                }

        return statistics
