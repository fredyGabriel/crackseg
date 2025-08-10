"""Statistical analysis utilities for experiment comparison."""

import logging
from typing import Any

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis utilities for experiment comparison."""

    def __init__(self, significance_threshold: float = 0.05) -> None:
        """Initialize the statistical analyzer.

        Args:
            significance_threshold: Threshold for statistical significance tests
        """
        self.logger = logging.getLogger(__name__)
        self.significance_threshold = significance_threshold

    def perform_statistical_analysis(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Perform statistical analysis on experiment metrics.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing comprehensive statistical analysis
        """
        analysis = {
            "descriptive_statistics": {},
            "correlation_analysis": {},
            "significance_tests": {},
        }

        # Calculate descriptive statistics for each metric
        for metric_name in ["iou", "dice", "f1", "precision", "recall"]:
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

    def _calculate_metric_correlations(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate correlations between different metrics.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary of metric correlation coefficients
        """
        correlations = {}
        metrics = ["iou", "dice", "f1", "precision", "recall"]

        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i + 1 :]:
                values1 = [
                    metrics.get(metric1, 0.0)
                    for metrics in metrics_data.values()
                ]
                values2 = [
                    metrics.get(metric2, 0.0)
                    for metrics in metrics_data.values()
                ]

                if len(values1) > 1 and len(values2) > 1:
                    correlation, _ = stats.pearsonr(values1, values2)
                    correlations[f"{metric1}_vs_{metric2}"] = correlation

        return correlations

    def _perform_significance_tests(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Perform statistical significance tests on experiment metrics.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing significance test results
        """
        significance_results = {}
        metrics = ["iou", "dice", "f1", "precision", "recall"]

        for metric in metrics:
            values = [
                metrics.get(metric, 0.0) for metrics in metrics_data.values()
            ]

            if len(values) > 2:
                # Perform Shapiro-Wilk test for normality
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    significance_results[metric] = {
                        "shapiro_wilk": {
                            "statistic": shapiro_stat,
                            "p_value": shapiro_p,
                            "is_normal": shapiro_p
                            > self.significance_threshold,
                        }
                    }
                except Exception as e:
                    self.logger.warning(
                        f"Shapiro-Wilk test failed for {metric}: {e}"
                    )

        return significance_results

    def check_statistical_significance(
        self, best_exp_id: str, composite_scores: dict[str, dict[str, Any]]
    ) -> bool:
        """Check if the best performing experiment is statistically significant.

        Args:
            best_exp_id: ID of the best performing experiment
            composite_scores: Dictionary of composite scores for all experiments

        Returns:
            True if the difference is statistically significant
        """
        if len(composite_scores) < 2:
            return False

        best_score = composite_scores[best_exp_id]["total_score"]
        other_scores = [
            score_data["total_score"]
            for exp_id, score_data in composite_scores.items()
            if exp_id != best_exp_id
        ]

        if not other_scores:
            return False

        # Perform one-sample t-test
        try:
            result = stats.ttest_1samp(other_scores, best_score)
            p_value = result.pvalue
            return p_value < self.significance_threshold
        except Exception as e:
            self.logger.warning(f"Statistical significance test failed: {e}")
            return False

    def calculate_confidence_level(
        self, best_exp_id: str, composite_scores: dict[str, dict[str, Any]]
    ) -> float:
        """Calculate confidence level for the best performing experiment.

        Args:
            best_exp_id: ID of the best performing experiment
            composite_scores: Dictionary of composite scores for all experiments

        Returns:
            Confidence level as a percentage (0-100)
        """
        if len(composite_scores) < 2:
            return 100.0

        best_score = composite_scores[best_exp_id]["total_score"]
        other_scores = [
            score_data["total_score"]
            for exp_id, score_data in composite_scores.items()
            if exp_id != best_exp_id
        ]

        if not other_scores:
            return 100.0

        # Calculate confidence interval
        try:
            mean_others = np.mean(other_scores)
            std_others = np.std(other_scores, ddof=1)

            if std_others == 0:
                return 100.0 if best_score > mean_others else 0.0

            # Calculate z-score
            z_score = (best_score - mean_others) / std_others

            # Convert to confidence level (assuming normal distribution)
            confidence = float(stats.norm.cdf(z_score)) * 100
            return min(max(confidence, 0.0), 100.0)
        except Exception as e:
            self.logger.warning(f"Confidence level calculation failed: {e}")
            return 50.0  # Default to 50% confidence
