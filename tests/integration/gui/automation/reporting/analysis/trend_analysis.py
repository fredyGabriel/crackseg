"""
Trend analysis for stakeholder reporting.

This module provides trend analysis capabilities for performance and quality
metrics over time, including statistical calculations and predictions.
"""

from typing import Any

import numpy as np


class PerformanceAnalyzer:
    """Analyzes performance trends and metrics."""

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze performance data and return trends."""
        page_load_times = data.get("page_load_times", {})
        memory_usage = data.get("memory_usage", {})

        return {
            "load_time_trends": {
                "current_avg": page_load_times.get("avg", 0.0),
                "trend_direction": self._determine_trend_direction(
                    page_load_times.get("avg", 0.0)
                ),
                "improvement_percentage": self._calculate_improvement(
                    page_load_times.get("avg", 0.0)
                ),
                "variability_coefficient": self._calculate_variability(
                    page_load_times
                ),
                "performance_stability": self._assess_stability(
                    page_load_times
                ),
            },
            "memory_usage_trends": {
                "current_avg": memory_usage.get("avg_mb", 0.0),
                "trend_direction": self._determine_trend_direction(
                    memory_usage.get("avg_mb", 0.0), reverse=True
                ),
                "efficiency_score": self._calculate_memory_efficiency(
                    memory_usage
                ),
            },
            "compliance_trends": {
                "page_load_compliance": data.get(
                    "page_load_compliance", False
                ),
                "config_validation_compliance": data.get(
                    "config_validation_compliance", False
                ),
            },
            "overall_performance_score": self._calculate_performance_score(
                data
            ),
        }

    def _determine_trend_direction(
        self, value: float, reverse: bool = False
    ) -> str:
        """Determine trend direction based on value."""
        if reverse:
            if value < 200:
                return "improving"
            elif value < 300:
                return "stable"
            else:
                return "degrading"
        else:
            if value < 1.0:
                return "improving"
            elif value < 2.0:
                return "stable"
            else:
                return "degrading"

    def _calculate_improvement(self, value: float) -> float:
        """Calculate improvement percentage."""
        baseline = 2.0  # 2 second baseline
        if value >= baseline:
            return 0.0
        return ((baseline - value) / baseline) * 100

    def _calculate_variability(self, data: dict[str, Any]) -> float:
        """Calculate coefficient of variation."""
        values = [
            data.get("min", 0),
            data.get("avg", 0),
            data.get("max", 0),
        ]
        if not any(values):
            return 0.0
        return float(
            np.std(values) / np.mean(values) if np.mean(values) > 0 else 0.0
        )

    def _assess_stability(self, data: dict[str, Any]) -> str:
        """Assess performance stability."""
        cv = self._calculate_variability(data)
        if cv < 0.1:
            return "high"
        elif cv < 0.3:
            return "medium"
        else:
            return "low"

    def _calculate_memory_efficiency(self, data: dict[str, Any]) -> float:
        """Calculate memory efficiency score."""
        avg_mb = data.get("avg_mb", 0)
        if avg_mb < 200:
            return 100.0
        elif avg_mb < 400:
            return 80.0
        elif avg_mb < 600:
            return 60.0
        else:
            return 40.0

    def _calculate_performance_score(self, data: dict[str, Any]) -> float:
        """Calculate overall performance score."""
        page_load_times = data.get("page_load_times", {})
        avg_load_time = page_load_times.get("avg", 2.0)

        # Score based on load time (lower is better)
        if avg_load_time < 1.0:
            return 100.0
        elif avg_load_time < 1.5:
            return 90.0
        elif avg_load_time < 2.0:
            return 80.0
        elif avg_load_time < 3.0:
            return 60.0
        else:
            return 40.0


class QualityAnalyzer:
    """Analyzes quality trends and metrics."""

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze quality data and return trends."""
        return {
            "success_rate_trends": {
                "current_rate": data.get("workflow_scenarios", {}).get(
                    "success_rate", 0.0
                ),
                "trend_direction": self._determine_quality_trend(
                    data.get("workflow_scenarios", {}).get("success_rate", 0.0)
                ),
            },
            "stability_trends": {
                "current_rate": data.get("concurrent_operations", {}).get(
                    "stability_rate", 0.0
                ),
                "trend_direction": self._determine_quality_trend(
                    data.get("concurrent_operations", {}).get(
                        "stability_rate", 0.0
                    )
                ),
            },
            "overall_quality_score": self._calculate_quality_score(data),
            "critical_areas": self._identify_critical_areas(data),
        }

    def _determine_quality_trend(self, rate: float) -> str:
        """Determine quality trend direction."""
        if rate >= 95:
            return "excellent"
        elif rate >= 90:
            return "good"
        elif rate >= 80:
            return "stable"
        else:
            return "degrading"

    def _calculate_quality_score(self, data: dict[str, Any]) -> float:
        """Calculate overall quality score."""
        scores = []

        # Workflow success rate
        workflow_rate = data.get("workflow_scenarios", {}).get(
            "success_rate", 0.0
        )
        scores.append(workflow_rate)

        # Error recovery rate
        error_rate = data.get("error_scenarios", {}).get(
            "error_recovery_rate", 0.0
        )
        scores.append(error_rate)

        # Session persistence rate
        persistence_rate = data.get("session_state", {}).get(
            "persistence_rate", 0.0
        )
        scores.append(persistence_rate)

        # Concurrent stability rate
        stability_rate = data.get("concurrent_operations", {}).get(
            "stability_rate", 0.0
        )
        scores.append(stability_rate)

        # Automation success rate
        automation_rate = data.get("automation_metrics", {}).get(
            "automation_success_rate", 0.0
        )
        scores.append(automation_rate)

        # Cleanup effectiveness rate
        cleanup_rate = data.get("resource_cleanup", {}).get(
            "cleanup_effectiveness_rate", 0.0
        )
        scores.append(cleanup_rate)

        return float(np.mean(scores) if scores else 0.0)

    def _identify_critical_areas(self, data: dict[str, Any]) -> list[str]:
        """Identify critical areas requiring attention."""
        critical_areas = []

        # Check workflow scenarios
        workflow_rate = data.get("workflow_scenarios", {}).get(
            "success_rate", 100.0
        )
        if workflow_rate < 90:
            critical_areas.append("workflow_scenarios")

        # Check error scenarios
        error_rate = data.get("error_scenarios", {}).get(
            "error_recovery_rate", 100.0
        )
        if error_rate < 85:
            critical_areas.append("error_scenarios")

        # Check concurrent operations
        stability_rate = data.get("concurrent_operations", {}).get(
            "stability_rate", 100.0
        )
        if stability_rate < 90:
            critical_areas.append("concurrent_operations")

        return critical_areas


class PredictionEngine:
    """Provides trend prediction capabilities."""

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict future trends based on current data."""
        return {
            "performance_predictions": {
                "load_time_forecast": self._predict_load_time(data),
                "memory_usage_forecast": self._predict_memory_usage(data),
            },
            "quality_predictions": {
                "success_rate_forecast": self._predict_success_rate(data),
                "stability_forecast": self._predict_stability(data),
            },
            "confidence_scores": {
                "overall_confidence": 0.85,
                "data_quality_score": 0.90,
            },
        }

    def _predict_load_time(self, data: dict[str, Any]) -> float:
        """Predict future load time."""
        current_avg = data.get("page_load_times", {}).get("avg", 1.5)
        # Simple prediction: slight improvement
        return max(0.5, current_avg * 0.95)

    def _predict_memory_usage(self, data: dict[str, Any]) -> float:
        """Predict future memory usage."""
        current_avg = data.get("memory_usage", {}).get("avg_mb", 250.0)
        # Simple prediction: slight increase
        return current_avg * 1.05

    def _predict_success_rate(self, data: dict[str, Any]) -> float:
        """Predict future success rate."""
        current_rate = data.get("workflow_scenarios", {}).get(
            "success_rate", 95.0
        )
        # Simple prediction: slight improvement
        return min(100.0, current_rate * 1.02)

    def _predict_stability(self, data: dict[str, Any]) -> float:
        """Predict future stability rate."""
        current_rate = data.get("concurrent_operations", {}).get(
            "stability_rate", 95.0
        )
        # Simple prediction: slight improvement
        return min(100.0, current_rate * 1.01)


class TrendAnalysisEngine:
    """Main engine for trend analysis combining performance and quality."""

    def __init__(self) -> None:
        """Initialize the trend analysis engine."""
        self.performance_analyzer = PerformanceAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.prediction_engine = PredictionEngine()

    def analyze_performance_trends(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze performance trends."""
        return self.performance_analyzer.analyze(data)

    def analyze_quality_trends(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze quality trends."""
        return self.quality_analyzer.analyze(data)

    def predict_future_trends(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict future trends."""
        return self.prediction_engine.predict(data)

    def analyze_comprehensive_trends(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze comprehensive trends combining performance and quality."""
        performance_trends = self.analyze_performance_trends(data)
        quality_trends = self.analyze_quality_trends(data)

        # Calculate combined health score
        perf_score = performance_trends.get("overall_performance_score", 0.0)
        quality_score = quality_trends.get("overall_quality_score", 0.0)
        combined_health_score = (perf_score + quality_score) / 2

        # Generate recommendations
        recommendations = self._generate_recommendations(data)

        return {
            "performance_trends": performance_trends,
            "quality_trends": quality_trends,
            "combined_health_score": combined_health_score,
            "recommendations": recommendations,
        }

    def _generate_recommendations(self, data: dict[str, Any]) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Performance recommendations
        performance_data = data.get("performance_metrics", {})
        if not performance_data.get("page_load_compliance", True):
            recommendations.append("Optimize page load performance")

        # Quality recommendations
        workflow_rate = data.get("workflow_scenarios", {}).get(
            "success_rate", 100.0
        )
        if workflow_rate < 95:
            recommendations.append("Improve workflow success rate")

        # Resource recommendations
        cleanup_rate = data.get("resource_cleanup", {}).get(
            "cleanup_effectiveness_rate", 100.0
        )
        if cleanup_rate < 95:
            recommendations.append("Enhance resource cleanup procedures")

        return recommendations
