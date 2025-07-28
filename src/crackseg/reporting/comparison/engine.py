"""Automated experiment comparison engine for crack segmentation.

This module provides a comprehensive comparison engine that automatically
analyzes multiple experiments, identifies best performers, and generates
detailed comparison reports with statistical analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..config import ExperimentData, ReportConfig
from ..interfaces import ComparisonEngine


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

        # Define comparison metrics and their weights
        self.metrics_config = {
            "iou": {"weight": 0.4, "higher_better": True},
            "dice": {"weight": 0.3, "higher_better": True},
            "f1": {"weight": 0.2, "higher_better": True},
            "precision": {"weight": 0.05, "higher_better": True},
            "recall": {"weight": 0.05, "higher_better": True},
            "loss": {
                "weight": 0.0,
                "higher_better": False,
            },  # Excluded from ranking
        }

        # Statistical significance threshold
        self.significance_threshold = 0.05

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

        # Perform statistical analysis
        comparison_results["statistical_analysis"] = (
            self._perform_statistical_analysis(metrics_data)
        )

        # Generate ranking analysis
        comparison_results["ranking_analysis"] = (
            self._generate_ranking_analysis(metrics_data)
        )

        # Analyze performance trends
        comparison_results["performance_trends"] = (
            self._analyze_performance_trends(experiments_data)
        )

        # Detect anomalies
        comparison_results["anomaly_detection"] = self._detect_anomalies(
            metrics_data
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
        Identify the best performing experiment using multi-criteria analysis.

        Args:
            experiments_data: List of experiment data
            config: Reporting configuration

        Returns:
            Dictionary with best performer analysis
        """
        self.logger.info("Identifying best performing experiment")

        if not experiments_data:
            return {"error": "No experiments to analyze"}

        # Extract metrics
        metrics_data = self._extract_comparison_metrics(experiments_data)

        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(metrics_data)

        # Identify best performer
        best_experiment_id = max(
            composite_scores.keys(),
            key=lambda x: composite_scores[x]["total_score"],
        )

        best_performer = {
            "experiment_id": best_experiment_id,
            "composite_score": composite_scores[best_experiment_id][
                "total_score"
            ],
            "ranking_position": 1,
            "score_breakdown": composite_scores[best_experiment_id],
            "statistical_significance": self._check_statistical_significance(
                best_experiment_id, composite_scores
            ),
            "confidence_level": self._calculate_confidence_level(
                best_experiment_id, composite_scores
            ),
        }

        # Add runner-up analysis
        sorted_experiments = sorted(
            composite_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True,
        )

        if len(sorted_experiments) > 1:
            runner_up_id = sorted_experiments[1][0]
            best_performer["runner_up"] = {
                "experiment_id": runner_up_id,
                "score_difference": (
                    best_performer["composite_score"]
                    - composite_scores[runner_up_id]["total_score"]
                ),
                "percentage_improvement": (
                    (
                        best_performer["composite_score"]
                        - composite_scores[runner_up_id]["total_score"]
                    )
                    / composite_scores[runner_up_id]["total_score"]
                    * 100
                ),
            }

        return best_performer

    def generate_comparison_table(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Generate comprehensive comparison table.

        Args:
            experiments_data: List of experiment data
            config: Reporting configuration

        Returns:
            Dictionary with comparison table data
        """
        self.logger.info("Generating comparison table")

        if not experiments_data:
            return {"error": "No experiments to compare"}

        # Extract metrics
        metrics_data = self._extract_comparison_metrics(experiments_data)

        # Create comparison table
        comparison_table = []

        for exp_id, metrics in metrics_data.items():
            row = {
                "experiment_id": exp_id,
                "model_config": self._extract_model_config(
                    exp_id, experiments_data
                ),
                "training_config": self._extract_training_config(
                    exp_id, experiments_data
                ),
                **metrics,
            }
            comparison_table.append(row)

        # Calculate statistics
        table_stats = self._calculate_table_statistics(metrics_data)

        return {
            "table_data": comparison_table,
            "statistics": table_stats,
            "table_format": "pandas_dataframe",
            "export_formats": ["csv", "excel", "json"],
        }

    def _extract_comparison_metrics(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, dict[str, float]]:
        """Extract comparison metrics from experiment data."""
        metrics_data = {}

        for exp_data in experiments_data:
            exp_id = exp_data.experiment_id
            metrics = {}

            # Extract final metrics from complete summary
            if "complete_summary" in exp_data.metrics:
                summary = exp_data.metrics["complete_summary"]
                best_metrics = summary.get("best_metrics", {})

                for metric_name in self.metrics_config.keys():
                    if metric_name in best_metrics:
                        metrics[metric_name] = best_metrics[metric_name].get(
                            "value", 0.0
                        )
                    else:
                        metrics[metric_name] = 0.0

            metrics_data[exp_id] = metrics

        return metrics_data

    def _perform_statistical_analysis(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Perform statistical analysis on experiment metrics."""
        analysis = {
            "descriptive_statistics": {},
            "correlation_analysis": {},
            "significance_tests": {},
        }

        # Calculate descriptive statistics for each metric
        for metric_name in self.metrics_config.keys():
            if metric_name == "loss":  # Skip loss for ranking
                continue

            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if values:
                analysis["descriptive_statistics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                }

        # Perform correlation analysis
        if len(metrics_data) > 2:
            analysis["correlation_analysis"] = (
                self._calculate_metric_correlations(metrics_data)
            )

        # Perform significance tests
        if len(metrics_data) > 2:
            analysis["significance_tests"] = self._perform_significance_tests(
                metrics_data
            )

        return analysis

    def _generate_ranking_analysis(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Generate comprehensive ranking analysis."""
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(metrics_data)

        # Create ranking
        ranking = sorted(
            composite_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True,
        )

        ranking_analysis = {
            "ranking": [
                {
                    "position": i + 1,
                    "experiment_id": exp_id,
                    "total_score": score_data["total_score"],
                    "score_breakdown": score_data,
                }
                for i, (exp_id, score_data) in enumerate(ranking)
            ],
            "score_distribution": {
                "mean": np.mean([s["total_score"] for _, s in ranking]),
                "std": np.std([s["total_score"] for _, s in ranking]),
                "min": np.min([s["total_score"] for _, s in ranking]),
                "max": np.max([s["total_score"] for _, s in ranking]),
            },
            "ranking_method": "weighted_composite_score",
            "weights_used": self.metrics_config,
        }

        return ranking_analysis

    def _calculate_composite_scores(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, Any]]:
        """Calculate composite scores for ranking."""
        composite_scores = {}

        for exp_id, metrics in metrics_data.items():
            score_breakdown = {}
            total_score = 0.0

            for metric_name, config in self.metrics_config.items():
                if metric_name == "loss":  # Skip loss
                    continue

                metric_value = metrics.get(metric_name, 0.0)
                weight = config["weight"]

                # Normalize score (0-1 range)
                normalized_score = min(max(metric_value, 0.0), 1.0)

                score_breakdown[metric_name] = {
                    "raw_value": metric_value,
                    "normalized_score": normalized_score,
                    "weight": weight,
                    "weighted_score": normalized_score * weight,
                }

                total_score += normalized_score * weight

            composite_scores[exp_id] = {
                "total_score": total_score,
                "score_breakdown": score_breakdown,
            }

        return composite_scores

    def _analyze_performance_trends(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Analyze performance trends across experiments."""
        trends = {
            "metric_trends": {},
            "improvement_patterns": {},
            "consistency_analysis": {},
        }

        # Analyze trends for each metric
        for metric_name in self.metrics_config.keys():
            if metric_name == "loss":
                continue

            values = []
            for exp_data in experiments_data:
                if "complete_summary" in exp_data.metrics:
                    summary = exp_data.metrics["complete_summary"]
                    best_metrics = summary.get("best_metrics", {})
                    if metric_name in best_metrics:
                        values.append(
                            best_metrics[metric_name].get("value", 0.0)
                        )
                    else:
                        values.append(0.0)

            if len(values) > 1:
                trends["metric_trends"][metric_name] = {
                    "values": values,
                    "trend_direction": (
                        "increasing"
                        if values[-1] > values[0]
                        else "decreasing"
                    ),
                    "improvement_rate": (
                        (values[-1] - values[0]) / len(values)
                        if len(values) > 1
                        else 0.0
                    ),
                    "consistency": np.std(values),
                }

        return trends

    def _detect_anomalies(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Detect anomalies in experiment results."""
        anomalies = {
            "outliers": {},
            "performance_gaps": {},
            "unusual_patterns": [],
        }

        for metric_name in self.metrics_config.keys():
            if metric_name == "loss":
                continue

            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if len(values) > 2:
                # Detect outliers using IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = [
                    (exp_id, value)
                    for exp_id, value in zip(
                        metrics_data.keys(), values, strict=False
                    )
                    if value < lower_bound or value > upper_bound
                ]

                if outliers:
                    anomalies["outliers"][metric_name] = outliers

        return anomalies

    def _generate_comparison_recommendations(
        self,
        metrics_data: dict[str, dict[str, float]],
        comparison_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on comparison analysis."""
        recommendations = []

        # Analyze best performer characteristics
        if "ranking_analysis" in comparison_results:
            ranking = comparison_results["ranking_analysis"]["ranking"]
            if ranking:
                best_experiment = ranking[0]
                recommendations.append(
                    f"Best performer: {best_experiment['experiment_id']} "
                    f"(Score: {best_experiment['total_score']:.4f})"
                )

        # Analyze performance gaps
        if "statistical_analysis" in comparison_results:
            stats = comparison_results["statistical_analysis"][
                "descriptive_statistics"
            ]
            for metric, stat_data in stats.items():
                if stat_data["std"] > 0.1:  # High variance
                    recommendations.append(
                        f"High variance in {metric} suggests need for "
                        "hyperparameter tuning"
                    )

        # Analyze trends
        if "performance_trends" in comparison_results:
            trends = comparison_results["performance_trends"]["metric_trends"]
            for metric, trend_data in trends.items():
                if trend_data["trend_direction"] == "increasing":
                    recommendations.append(
                        f"Positive trend in {metric} - "
                        "continue current approach"
                    )
                elif trend_data["trend_direction"] == "decreasing":
                    recommendations.append(
                        f"Negative trend in {metric} - investigate issues"
                    )

        return recommendations

    def _extract_model_config(
        self, exp_id: str, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Extract model configuration for an experiment."""
        for exp_data in experiments_data:
            if exp_data.experiment_id == exp_id:
                return exp_data.config.get("model", {})
        return {}

    def _extract_training_config(
        self, exp_id: str, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Extract training configuration for an experiment."""
        for exp_data in experiments_data:
            if exp_data.experiment_id == exp_id:
                return exp_data.config.get("training", {})
        return {}

    def _calculate_table_statistics(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Calculate statistics for comparison table."""
        stats = {}

        for metric_name in self.metrics_config.keys():
            if metric_name == "loss":
                continue

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
                }

        return stats

    def _calculate_metric_correlations(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate correlations between metrics."""
        correlations = {}

        metric_names = [
            name for name in self.metrics_config.keys() if name != "loss"
        ]

        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i + 1 :]:
                values1 = [
                    metrics.get(metric1, 0.0)
                    for metrics in metrics_data.values()
                ]
                values2 = [
                    metrics.get(metric2, 0.0)
                    for metrics in metrics_data.values()
                ]

                if len(values1) > 1 and len(values2) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{metric1}_vs_{metric2}"] = correlation

        return correlations

    def _perform_significance_tests(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Perform statistical significance tests."""
        significance_tests = {}

        for metric_name in self.metrics_config.keys():
            if metric_name == "loss":
                continue

            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if len(values) > 2:
                # Perform one-sample t-test against zero
                result = stats.ttest_1samp(values, 0)
                t_stat = float(result.statistic)
                p_value = float(result.pvalue)
                significance_tests[metric_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < self.significance_threshold,
                }

        return significance_tests

    def _check_statistical_significance(
        self, best_exp_id: str, composite_scores: dict[str, dict[str, Any]]
    ) -> bool:
        """Check if best performer is statistically significant."""
        scores = [data["total_score"] for data in composite_scores.values()]
        if len(scores) < 2:
            return False

        best_score = composite_scores[best_exp_id]["total_score"]
        other_scores = [s for s in scores if s != best_score]

        if not other_scores:
            return True

        # Perform t-test
        result = stats.ttest_1samp(other_scores, best_score)
        p_value = float(result.pvalue)
        return p_value < self.significance_threshold

    def _calculate_confidence_level(
        self, best_exp_id: str, composite_scores: dict[str, dict[str, Any]]
    ) -> float:
        """Calculate confidence level for best performer."""
        scores = [data["total_score"] for data in composite_scores.values()]
        if len(scores) < 2:
            return 1.0

        best_score = composite_scores[best_exp_id]["total_score"]
        other_scores = [s for s in scores if s != best_score]

        if not other_scores:
            return 1.0

        # Calculate confidence based on score separation
        mean_other = np.mean(other_scores)
        std_other = np.std(other_scores)

        if std_other == 0:
            return 1.0 if best_score > mean_other else 0.0

        z_score = (best_score - mean_other) / std_other
        confidence = 1 - stats.norm.cdf(-z_score)

        return max(0.0, min(1.0, float(confidence)))
