"""Ranking and scoring analysis for experiment comparison."""

import logging
from typing import Any

import numpy as np


class RankingAnalyzer:
    """Ranking and scoring analysis utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the ranking analyzer."""
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

    def generate_ranking_analysis(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Generate comprehensive ranking analysis.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing comprehensive ranking analysis
        """
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
        """Calculate composite scores for ranking.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary mapping experiment IDs to composite score data
        """
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

                # Apply weight
                weighted_score = normalized_score * weight

                score_breakdown[metric_name] = {
                    "raw_value": metric_value,
                    "normalized_value": normalized_score,
                    "weight": weight,
                    "weighted_score": weighted_score,
                }

                total_score += weighted_score

            composite_scores[exp_id] = {
                "total_score": total_score,
                "score_breakdown": score_breakdown,
                "rank": None,  # Will be set during ranking
            }

        return composite_scores

    def identify_best_performer(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Identify the best performing experiment.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing best performer analysis
        """
        if not metrics_data:
            return {"error": "No experiments to analyze"}

        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(metrics_data)

        # Find best performer
        best_exp_id = max(
            composite_scores.items(), key=lambda x: x[1]["total_score"]
        )[0]

        best_score_data = composite_scores[best_exp_id]

        # Calculate performance gap
        other_scores = [
            score_data["total_score"]
            for exp_id, score_data in composite_scores.items()
            if exp_id != best_exp_id
        ]

        performance_gap = 0.0
        if other_scores:
            best_score = best_score_data["total_score"]
            second_best = max(other_scores)
            performance_gap = best_score - second_best

        return {
            "best_experiment_id": best_exp_id,
            "best_score": best_score_data["total_score"],
            "score_breakdown": best_score_data["score_breakdown"],
            "performance_gap": performance_gap,
            "total_experiments": len(composite_scores),
            "ranking_position": 1,
        }

    def get_performance_summary(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Get a summary of performance across all experiments.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing performance summary
        """
        if not metrics_data:
            return {"error": "No experiments to analyze"}

        summary = {
            "total_experiments": len(metrics_data),
            "metrics_summary": {},
            "overall_performance": {},
        }

        # Analyze each metric
        for metric_name in ["iou", "dice", "f1", "precision", "recall"]:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if values:
                summary["metrics_summary"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "best_experiment": max(
                        metrics_data.items(),
                        key=lambda x: x[1].get(metric_name, 0.0),
                    )[0],
                }

        # Overall performance analysis
        composite_scores = self._calculate_composite_scores(metrics_data)
        all_scores = [
            score_data["total_score"]
            for score_data in composite_scores.values()
        ]

        summary["overall_performance"] = {
            "mean_score": np.mean(all_scores),
            "std_score": np.std(all_scores),
            "min_score": np.min(all_scores),
            "max_score": np.max(all_scores),
            "score_range": np.max(all_scores) - np.min(all_scores),
        }

        return summary
