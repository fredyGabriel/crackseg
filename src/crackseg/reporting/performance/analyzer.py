"""Main performance analyzer for experiment evaluation.

This module provides the core ExperimentPerformanceAnalyzer class that
orchestrates all performance analysis components.
"""

import logging
from typing import Any

from ..config import ExperimentData, ReportConfig
from ..interfaces import PerformanceAnalyzer as PerformanceAnalyzerInterface
from .anomaly_detector import AnomalyDetector
from .metric_evaluator import MetricEvaluator
from .recommendation_engine import RecommendationEngine
from .training_analyzer import TrainingAnalyzer


class ExperimentPerformanceAnalyzer(PerformanceAnalyzerInterface):
    """
    PerformanceAnalyzer implementation for detailed experiment analysis.

    This class provides comprehensive performance analysis including:
    - Metric evaluation against thresholds
    - Anomaly detection across experiments
    - Training pattern analysis
    - Actionable recommendations generation
    """

    def __init__(self) -> None:
        """Initialize the ExperimentPerformanceAnalyzer."""
        self.logger = logging.getLogger(__name__)

        # Initialize specialized components
        self.metric_evaluator = MetricEvaluator()
        self.anomaly_detector = AnomalyDetector()
        self.training_analyzer = TrainingAnalyzer()
        self.recommendation_engine = RecommendationEngine()

    def analyze_performance(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Analyze performance metrics and generate insights.

        Args:
            experiment_data: Loaded experiment data
            config: Reporting configuration

        Returns:
            Dictionary with performance analysis results
        """
        self.logger.info(
            f"Analyzing performance for experiment: "
            f"{experiment_data.experiment_id}"
        )

        analysis = {
            "experiment_id": experiment_data.experiment_id,
            "analysis_timestamp": experiment_data.metadata.get(
                "generation_timestamp", "unknown"
            ),
            "metric_evaluation": {},
            "performance_score": 0.0,
            "threshold_compliance": {},
            "training_analysis": {},
            "insights": [],
            "warnings": [],
        }

        # Analyze complete summary metrics
        if "complete_summary" in experiment_data.metrics:
            complete_summary = experiment_data.metrics["complete_summary"]
            analysis["metric_evaluation"] = (
                self.metric_evaluator.evaluate_metrics(
                    complete_summary, config
                )
            )
            analysis["performance_score"] = (
                self.metric_evaluator.calculate_performance_score(
                    complete_summary, config
                )
            )
            analysis["threshold_compliance"] = (
                self.metric_evaluator.check_threshold_compliance(
                    complete_summary, config
                )
            )

        # Analyze training patterns
        if "epoch_metrics" in experiment_data.metrics:
            epoch_metrics = experiment_data.metrics["epoch_metrics"]
            analysis["training_analysis"] = (
                self.training_analyzer.analyze_training_patterns(epoch_metrics)
            )

        # Generate insights
        analysis["insights"] = self._generate_insights(analysis)

        # Generate warnings
        analysis["warnings"] = self._generate_warnings(analysis)

        self.logger.info(
            f"Performance analysis completed for "
            f"{experiment_data.experiment_id}"
        )
        return analysis

    def detect_anomalies(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Detect performance anomalies across experiments.

        Args:
            experiments_data: List of experiment data
            config: Reporting configuration

        Returns:
            Dictionary with anomaly detection results
        """
        return self.anomaly_detector.detect_anomalies(experiments_data, config)

    def generate_recommendations(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> list[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            experiment_data: Loaded experiment data
            config: Reporting configuration

        Returns:
            List of actionable recommendations
        """
        return self.recommendation_engine.generate_recommendations(
            experiment_data, config
        )

    def _generate_insights(self, analysis: dict[str, Any]) -> list[str]:
        """Generate insights from analysis results."""
        insights = []

        # Performance insights
        performance_score = analysis.get("performance_score", 0.0)
        if performance_score >= 0.9:
            insights.append("Excellent overall performance across all metrics")
        elif performance_score >= 0.7:
            insights.append("Good performance with room for improvement")
        else:
            insights.append(
                "Performance below expectations, significant improvements "
                "needed"
            )

        # Training insights
        training_analysis = analysis.get("training_analysis", {})
        convergence = training_analysis.get("convergence_analysis", {})

        if convergence.get("converged", False):
            insights.append("Training converged successfully")
        else:
            insights.append(
                "Training may not have converged - consider longer training"
            )

        if convergence.get("loss_trend") == "increasing":
            insights.append(
                "Loss trend is increasing - potential training instability"
            )

        # Stability insights
        stability = training_analysis.get("training_stability", {})
        if stability.get("iou_stability") == "unstable":
            insights.append(
                "IoU shows high variance - consider regularization techniques"
            )

        if stability.get("overfitting_risk") == "high":
            insights.append(
                "High overfitting risk detected - consider early stopping or "
                "regularization"
            )

        return insights

    def _generate_warnings(self, analysis: dict[str, Any]) -> list[str]:
        """Generate warnings from analysis results."""
        warnings = []

        # Threshold compliance warnings
        compliance = analysis.get("threshold_compliance", {})
        if not compliance.get("iou_compliant", True):
            warnings.append("IoU below minimum threshold")
        if not compliance.get("f1_compliant", True):
            warnings.append("F1 score below minimum threshold")
        if not compliance.get("precision_compliant", True):
            warnings.append("Precision below minimum threshold")
        if not compliance.get("recall_compliant", True):
            warnings.append("Recall below minimum threshold")

        # Training warnings
        training_analysis = analysis.get("training_analysis", {})
        convergence = training_analysis.get("convergence_analysis", {})

        if not convergence.get("converged", True):
            warnings.append("Training may not have converged")

        if convergence.get("loss_trend") == "increasing":
            warnings.append(
                "Loss is increasing - training instability detected"
            )

        stability = training_analysis.get("training_stability", {})
        if stability.get("overfitting_risk") == "high":
            warnings.append("High overfitting risk detected")

        return warnings
