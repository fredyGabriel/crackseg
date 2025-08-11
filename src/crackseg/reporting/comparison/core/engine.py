"""Core comparison engine with modular architecture."""

import logging
from typing import Any

import pandas as pd

from ...config import ExperimentData, ReportConfig
from ...interfaces import ComparisonEngine
from ..analysis.anomalies import AnomalyDetector
from ..analysis.ranking import RankingAnalyzer
from ..analysis.statistical import StatisticalAnalyzer
from ..analysis.trends import TrendAnalyzer
from ..utils.metrics import MetricsExtractor
from ..utils.recommendations import RecommendationGenerator
from ..utils.table_utils import TableUtils


class AutomatedComparisonEngine(ComparisonEngine):
    """
    Automated experiment comparison engine with modular architecture.

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

        # Initialize modular components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.ranking_analyzer = RankingAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.metrics_extractor = MetricsExtractor()
        self.recommendation_generator = RecommendationGenerator()
        self.table_utils = TableUtils()

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
        metrics_data = self.metrics_extractor.extract_comparison_metrics(
            experiments_data
        )

        # Perform statistical analysis
        comparison_results["statistical_analysis"] = (
            self.statistical_analyzer.perform_statistical_analysis(
                metrics_data
            )
        )

        # Generate ranking analysis
        comparison_results["ranking_analysis"] = (
            self.ranking_analyzer.generate_ranking_analysis(metrics_data)
        )

        # Analyze performance trends
        comparison_results["performance_trends"] = (
            self.trend_analyzer.analyze_performance_trends(experiments_data)
        )

        # Detect anomalies
        comparison_results["anomaly_detection"] = (
            self.anomaly_detector.detect_anomalies(metrics_data)
        )

        # Generate recommendations
        comparison_results["recommendations"] = (
            self.recommendation_generator.generate_comparison_recommendations(
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
        Identify the best performing experiment.

        Args:
            experiments_data: List of experiment data to analyze
            config: Reporting configuration

        Returns:
            Dictionary containing best performer analysis
        """
        self.logger.info("Identifying best performing experiment")

        if len(experiments_data) < 1:
            return {"error": "No experiments to analyze"}

        # Extract metrics
        metrics_data = self.metrics_extractor.extract_comparison_metrics(
            experiments_data
        )

        # Identify best performer
        best_performer = self.ranking_analyzer.identify_best_performer(
            metrics_data
        )

        # Add statistical significance check
        if "best_experiment_id" in best_performer:
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
        Generate comparison table with statistics.

        Args:
            experiments_data: List of experiment data to compare
            config: Reporting configuration

        Returns:
            Dictionary containing comparison table data
        """
        self.logger.info("Generating comparison table")

        if len(experiments_data) < 1:
            return {"error": "No experiments to compare"}

        # Extract metrics
        metrics_data = self.metrics_extractor.extract_comparison_metrics(
            experiments_data
        )

        # Generate table data
        table_data = self.table_utils.format_comparison_table(metrics_data)

        # Add table summary
        table_summary = self.table_utils.generate_table_summary(metrics_data)

        return {
            "table_data": table_data,
            "table_summary": table_summary,
            "export_formats": ["dict", "list", "csv"],
        }

    def get_detailed_analysis(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Get detailed analysis including all components.

        Args:
            experiments_data: List of experiment data to analyze
            config: Reporting configuration

        Returns:
            Dictionary containing comprehensive analysis
        """
        self.logger.info("Generating detailed analysis")

        # Get basic comparison
        comparison_results = self.compare_experiments(experiments_data, config)

        # Extract metrics
        metrics_data = self.metrics_extractor.extract_comparison_metrics(
            experiments_data
        )

        # Add detailed analysis components
        detailed_analysis = {
            **comparison_results,
            "metrics_summary": self.metrics_extractor.get_metrics_summary(
                metrics_data
            ),
            "trend_summary": self.trend_analyzer.get_trend_summary(
                experiments_data
            ),
            "anomaly_summary": self.anomaly_detector.get_anomaly_summary(
                metrics_data
            ),
            "performance_summary": self.ranking_analyzer.get_performance_summary(
                metrics_data
            ),
            "recommendation_summary": self.recommendation_generator.generate_summary_recommendations(
                comparison_results
            ),
        }

        return detailed_analysis
