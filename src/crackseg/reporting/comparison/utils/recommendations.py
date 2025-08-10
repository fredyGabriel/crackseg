"""Recommendation generation for experiment comparison."""

import logging
from typing import Any


class RecommendationGenerator:
    """Recommendation generation utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the recommendation generator."""
        self.logger = logging.getLogger(__name__)

    def generate_comparison_recommendations(
        self,
        metrics_data: dict[str, dict[str, float]],
        comparison_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on comparison analysis.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries
            comparison_results: Dictionary containing comparison analysis results

        Returns:
            List of recommendation strings
        """
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

        # Add anomaly-based recommendations
        if "anomaly_detection" in comparison_results:
            anomalies = comparison_results["anomaly_detection"]
            if anomalies.get("suspicious_experiments"):
                recommendations.append(
                    f"Found {len(anomalies['suspicious_experiments'])} suspicious experiments - "
                    "review configurations and results"
                )

        # Add statistical significance recommendations
        if "statistical_analysis" in comparison_results:
            significance_tests = comparison_results[
                "statistical_analysis"
            ].get("significance_tests", {})
            if significance_tests:
                recommendations.append(
                    "Statistical significance tests performed - "
                    "review p-values for confidence in results"
                )

        return recommendations

    def generate_actionable_recommendations(
        self, comparison_results: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations for next steps.

        Args:
            comparison_results: Dictionary containing comparison analysis results

        Returns:
            List of actionable recommendation strings
        """
        actions = []

        # Check if we have enough experiments for meaningful comparison
        if "experiment_count" in comparison_results:
            count = comparison_results["experiment_count"]
            if count < 3:
                actions.append(
                    f"Only {count} experiments available - "
                    "run more experiments for better statistical analysis"
                )

        # Check for performance gaps
        if "ranking_analysis" in comparison_results:
            ranking = comparison_results["ranking_analysis"]["ranking"]
            if len(ranking) >= 2:
                best_score = ranking[0]["total_score"]
                second_score = ranking[1]["total_score"]
                gap = best_score - second_score

                if gap < 0.01:
                    actions.append(
                        "Small performance gap between top experiments - "
                        "consider ensemble methods or further tuning"
                    )
                elif gap > 0.1:
                    actions.append(
                        "Large performance gap detected - "
                        "investigate why top performer is significantly better"
                    )

        # Check for consistency issues
        if "performance_trends" in comparison_results:
            trends = comparison_results["performance_trends"]
            if "consistency_analysis" in trends:
                consistency = trends["consistency_analysis"]
                low_consistency = consistency.get("low_consistency", [])
                if low_consistency:
                    actions.append(
                        f"Low consistency detected in: {', '.join(low_consistency)} - "
                        "implement more robust evaluation protocols"
                    )

        return actions

    def generate_optimization_recommendations(
        self, comparison_results: dict[str, Any]
    ) -> list[str]:
        """Generate optimization-focused recommendations.

        Args:
            comparison_results: Dictionary containing comparison analysis results

        Returns:
            List of optimization recommendation strings
        """
        optimizations = []

        # Analyze metric-specific recommendations
        if "statistical_analysis" in comparison_results:
            stats = comparison_results["statistical_analysis"].get(
                "descriptive_statistics", {}
            )

            for metric, stat_data in stats.items():
                mean_val = stat_data.get("mean", 0)
                std_val = stat_data.get("std", 0)

                if mean_val < 0.5:
                    optimizations.append(
                        f"Low average {metric} ({mean_val:.3f}) - "
                        "focus on improving this metric"
                    )

                if std_val > 0.15:
                    optimizations.append(
                        f"High variance in {metric} ({std_val:.3f}) - "
                        "implement more consistent training protocols"
                    )

        # Analyze correlation insights
        if "statistical_analysis" in comparison_results:
            correlations = comparison_results["statistical_analysis"].get(
                "correlation_analysis", {}
            )

            for correlation_name, correlation_value in correlations.items():
                if abs(correlation_value) > 0.8:
                    optimizations.append(
                        f"Strong correlation in {correlation_name} ({correlation_value:.3f}) - "
                        "consider focusing on one metric or finding orthogonal improvements"
                    )

        return optimizations

    def generate_summary_recommendations(
        self, comparison_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a comprehensive summary of all recommendations.

        Args:
            comparison_results: Dictionary containing comparison analysis results

        Returns:
            Dictionary containing categorized recommendations
        """
        summary = {
            "general_recommendations": self.generate_comparison_recommendations(
                {}, comparison_results
            ),
            "actionable_items": self.generate_actionable_recommendations(
                comparison_results
            ),
            "optimization_suggestions": self.generate_optimization_recommendations(
                comparison_results
            ),
            "priority_levels": {
                "high": [],
                "medium": [],
                "low": [],
            },
        }

        # Categorize recommendations by priority
        all_recommendations = (
            summary["general_recommendations"]
            + summary["actionable_items"]
            + summary["optimization_suggestions"]
        )

        for rec in all_recommendations:
            if any(
                keyword in rec.lower()
                for keyword in ["error", "issue", "problem", "suspicious"]
            ):
                summary["priority_levels"]["high"].append(rec)
            elif any(
                keyword in rec.lower()
                for keyword in ["tuning", "optimization", "improvement"]
            ):
                summary["priority_levels"]["medium"].append(rec)
            else:
                summary["priority_levels"]["low"].append(rec)

        return summary
