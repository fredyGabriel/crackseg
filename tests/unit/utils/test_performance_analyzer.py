"""Unit tests for ExperimentPerformanceAnalyzer."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from crackseg.reporting.config import ExperimentData, ReportConfig
from crackseg.reporting.performance_analyzer import (
    ExperimentPerformanceAnalyzer,
)


class TestExperimentPerformanceAnalyzer:
    """Test ExperimentPerformanceAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self) -> ExperimentPerformanceAnalyzer:
        """Provide ExperimentPerformanceAnalyzer instance."""
        return ExperimentPerformanceAnalyzer()

    @pytest.fixture
    def config(self) -> ReportConfig:
        """Provide ReportConfig instance."""
        return ReportConfig()

    @pytest.fixture
    def sample_experiment_data(self) -> ExperimentData:
        """Provide sample experiment data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "test_experiment"
            experiment_dir.mkdir()

            return ExperimentData(
                experiment_id="test_experiment",
                experiment_dir=experiment_dir,
                config=Mock(),
                metrics={
                    "complete_summary": {
                        "best_iou": 0.78,
                        "best_f1": 0.82,
                        "best_precision": 0.85,
                        "best_recall": 0.79,
                        "final_loss": 0.12,
                        "training_time": 3600.5,
                        "best_epoch": 85,
                    },
                    "epoch_metrics": [
                        {"epoch": 1, "loss": 2.5, "iou": 0.3},
                        {"epoch": 2, "loss": 2.1, "iou": 0.4},
                        {"epoch": 3, "loss": 1.8, "iou": 0.5},
                        {"epoch": 4, "loss": 1.5, "iou": 0.6},
                        {"epoch": 5, "loss": 1.2, "iou": 0.7},
                    ],
                },
                artifacts={},
                metadata={"experiment_name": "test_experiment"},
            )

    @pytest.fixture
    def multiple_experiments_data(self) -> list[ExperimentData]:
        """Provide multiple experiment data for comparison."""
        experiments = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                experiment_dir = Path(temp_dir) / f"test_experiment_{i}"
                experiment_dir.mkdir()

                # Vary metrics slightly for anomaly detection
                base_iou = 0.75 + (i * 0.02)
                base_f1 = 0.80 + (i * 0.01)

                experiments.append(
                    ExperimentData(
                        experiment_id=f"test_experiment_{i}",
                        experiment_dir=experiment_dir,
                        config=Mock(),
                        metrics={
                            "complete_summary": {
                                "best_iou": base_iou,
                                "best_f1": base_f1,
                                "best_precision": 0.85,
                                "best_recall": 0.79,
                                "final_loss": 0.12 + (i * 0.01),
                                "training_time": 3600.5,
                                "best_epoch": 85,
                            },
                        },
                        artifacts={},
                        metadata={"experiment_name": f"test_experiment_{i}"},
                    )
                )

            return experiments

    def test_analyze_performance_success(
        self,
        analyzer: ExperimentPerformanceAnalyzer,
        sample_experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> None:
        """Test successful performance analysis."""
        analysis = analyzer.analyze_performance(sample_experiment_data, config)

        assert analysis["experiment_id"] == "test_experiment"
        assert "metric_evaluation" in analysis
        assert "performance_score" in analysis
        assert "threshold_compliance" in analysis
        assert "training_analysis" in analysis
        assert "insights" in analysis
        assert "warnings" in analysis

        # Check performance score
        assert 0.0 <= analysis["performance_score"] <= 1.0

        # Check threshold compliance
        compliance = analysis["threshold_compliance"]
        assert compliance["iou_compliant"] is True
        assert compliance["f1_compliant"] is True
        assert compliance["precision_compliant"] is True
        assert compliance["recall_compliant"] is True

    def test_analyze_performance_no_metrics(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test performance analysis with no metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "test_experiment_no_metrics"
            experiment_dir.mkdir()

            experiment_data = ExperimentData(
                experiment_id="test_experiment_no_metrics",
                experiment_dir=experiment_dir,
                config=Mock(),
                metrics={},
                artifacts={},
                metadata={},
            )

            analysis = analyzer.analyze_performance(experiment_data, config)

            assert analysis["experiment_id"] == "test_experiment_no_metrics"
            assert analysis["performance_score"] == 0.0
            # Insights and warnings may still be generated even without metrics
            assert "insights" in analysis
            assert "warnings" in analysis

    def test_detect_anomalies_success(
        self,
        analyzer: ExperimentPerformanceAnalyzer,
        multiple_experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> None:
        """Test successful anomaly detection."""
        anomalies = analyzer.detect_anomalies(
            multiple_experiments_data, config
        )

        assert anomalies["total_experiments"] == 3
        assert "anomalies_detected" in anomalies
        assert "anomaly_details" in anomalies
        assert "statistical_summary" in anomalies
        assert "outlier_experiments" in anomalies

        # Check statistical summary
        stats = anomalies["statistical_summary"]
        assert "iou" in stats
        assert "f1" in stats
        assert "loss" in stats

        # Check that all metrics have expected structure
        for metric in ["iou", "f1", "loss"]:
            assert "mean" in stats[metric]
            assert "std" in stats[metric]
            assert "min" in stats[metric]
            assert "max" in stats[metric]

    def test_detect_anomalies_single_experiment(
        self,
        analyzer: ExperimentPerformanceAnalyzer,
        sample_experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> None:
        """Test anomaly detection with single experiment."""
        anomalies = analyzer.detect_anomalies([sample_experiment_data], config)

        assert anomalies["total_experiments"] == 1
        assert anomalies["anomalies_detected"] == 0
        assert len(anomalies["anomaly_details"]) == 0

    def test_detect_anomalies_no_metrics(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test anomaly detection with experiments without metrics."""
        experiments = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(2):
                experiment_dir = (
                    Path(temp_dir) / f"test_experiment_no_metrics_{i}"
                )
                experiment_dir.mkdir()

                experiments.append(
                    ExperimentData(
                        experiment_id=f"test_experiment_no_metrics_{i}",
                        experiment_dir=experiment_dir,
                        config=Mock(),
                        metrics={},
                        artifacts={},
                        metadata={},
                    )
                )

            anomalies = analyzer.detect_anomalies(experiments, config)

            assert anomalies["total_experiments"] == 2
            assert anomalies["anomalies_detected"] == 0
            assert len(anomalies["anomaly_details"]) == 0

    def test_generate_recommendations_success(
        self,
        analyzer: ExperimentPerformanceAnalyzer,
        sample_experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> None:
        """Test successful recommendation generation."""
        recommendations = analyzer.generate_recommendations(
            sample_experiment_data, config
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check that recommendations are actionable
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

    def test_generate_recommendations_no_metrics(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test recommendation generation with no metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "test_experiment_no_metrics"
            experiment_dir.mkdir()

            experiment_data = ExperimentData(
                experiment_id="test_experiment_no_metrics",
                experiment_dir=experiment_dir,
                config=Mock(),
                metrics={},
                artifacts={},
                metadata={},
            )

            recommendations = analyzer.generate_recommendations(
                experiment_data, config
            )

            assert isinstance(recommendations, list)
            assert len(recommendations) == 1
            assert "No complete metrics available" in recommendations[0]

    def test_generate_recommendations_below_thresholds(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test recommendation generation for poor performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "test_experiment_poor"
            experiment_dir.mkdir()

            experiment_data = ExperimentData(
                experiment_id="test_experiment_poor",
                experiment_dir=experiment_dir,
                config=Mock(),
                metrics={
                    "complete_summary": {
                        "best_iou": 0.5,  # Below threshold
                        "best_f1": 0.6,  # Below threshold
                        "best_precision": 0.7,  # Below threshold
                        "best_recall": 0.6,  # Below threshold
                        "final_loss": 2.0,  # High loss
                        "training_time": 8000,  # High training time
                        "best_epoch": 85,
                    },
                },
                artifacts={},
                metadata={},
            )

            recommendations = analyzer.generate_recommendations(
                experiment_data, config
            )

            assert isinstance(recommendations, list)
            assert len(recommendations) > 0

            # Check that recommendations address poor performance
            recommendation_text = " ".join(recommendations).lower()
            assert (
                "iou" in recommendation_text
                or "threshold" in recommendation_text
            )
            assert (
                "f1" in recommendation_text or "score" in recommendation_text
            )

    def test_evaluate_metrics(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test metric evaluation."""
        complete_summary = {
            "best_iou": 0.78,
            "best_f1": 0.82,
            "best_precision": 0.85,
            "best_recall": 0.79,
        }

        evaluation = analyzer._evaluate_metrics(complete_summary, config)

        assert "iou_score" in evaluation
        assert "f1_score" in evaluation
        assert "precision_score" in evaluation
        assert "recall_score" in evaluation
        assert "overall_score" in evaluation
        assert "metric_quality" in evaluation

        # Check scores are normalized
        for score_key in [
            "iou_score",
            "f1_score",
            "precision_score",
            "recall_score",
        ]:
            assert 0.0 <= evaluation[score_key] <= 1.0

        # Check metric quality
        quality = evaluation["metric_quality"]
        assert "iou" in quality
        assert "f1" in quality
        assert "precision" in quality
        assert "recall" in quality

    def test_calculate_performance_score(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test performance score calculation."""
        complete_summary = {
            "best_iou": 0.78,
            "best_f1": 0.82,
            "best_precision": 0.85,
            "best_recall": 0.79,
        }

        score = analyzer._calculate_performance_score(complete_summary, config)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_check_threshold_compliance(
        self, analyzer: ExperimentPerformanceAnalyzer, config: ReportConfig
    ) -> None:
        """Test threshold compliance checking."""
        complete_summary = {
            "best_iou": 0.78,
            "best_f1": 0.82,
            "best_precision": 0.85,
            "best_recall": 0.79,
        }

        compliance = analyzer._check_threshold_compliance(
            complete_summary, config
        )

        assert "iou_compliant" in compliance
        assert "f1_compliant" in compliance
        assert "precision_compliant" in compliance
        assert "recall_compliant" in compliance

        # All should be compliant with good metrics
        assert all(compliance.values())

    def test_analyze_training_patterns(
        self, analyzer: ExperimentPerformanceAnalyzer
    ) -> None:
        """Test training pattern analysis."""
        epoch_metrics = [
            {"epoch": 1, "loss": 2.5, "iou": 0.3},
            {"epoch": 2, "loss": 2.1, "iou": 0.4},
            {"epoch": 3, "loss": 1.8, "iou": 0.5},
            {"epoch": 4, "loss": 1.5, "iou": 0.6},
            {"epoch": 5, "loss": 1.2, "iou": 0.7},
        ]

        analysis = analyzer._analyze_training_patterns(epoch_metrics)

        assert "total_epochs" in analysis
        assert "convergence_analysis" in analysis
        assert "training_stability" in analysis
        assert "learning_patterns" in analysis

        assert analysis["total_epochs"] == 5

        # Check convergence analysis
        convergence = analysis["convergence_analysis"]
        assert "loss_trend" in convergence
        assert "converged" in convergence

    def test_analyze_training_patterns_empty(
        self, analyzer: ExperimentPerformanceAnalyzer
    ) -> None:
        """Test training pattern analysis with empty data."""
        analysis = analyzer._analyze_training_patterns([])

        assert "error" in analysis
        assert "No epoch metrics available" in analysis["error"]

    def test_generate_insights(
        self, analyzer: ExperimentPerformanceAnalyzer
    ) -> None:
        """Test insight generation."""
        analysis = {
            "performance_score": 0.85,
            "training_analysis": {
                "convergence_analysis": {
                    "converged": True,
                    "loss_trend": "decreasing",
                },
                "training_stability": {
                    "iou_stability": "stable",
                    "overfitting_risk": "low",
                },
            },
        }

        insights = analyzer._generate_insights(analysis)

        assert isinstance(insights, list)
        assert len(insights) > 0

        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0

    def test_generate_warnings(
        self, analyzer: ExperimentPerformanceAnalyzer
    ) -> None:
        """Test warning generation."""
        analysis = {
            "threshold_compliance": {
                "iou_compliant": False,
                "f1_compliant": True,
                "precision_compliant": False,
                "recall_compliant": True,
            },
            "training_analysis": {
                "convergence_analysis": {
                    "converged": False,
                    "loss_trend": "increasing",
                },
                "training_stability": {
                    "overfitting_risk": "high",
                },
            },
        }

        warnings = analyzer._generate_warnings(analysis)

        assert isinstance(warnings, list)
        assert len(warnings) > 0

        # Check that warnings address the issues
        warning_text = " ".join(warnings).lower()
        assert "iou" in warning_text
        assert "precision" in warning_text
        assert "converged" in warning_text
        assert "increasing" in warning_text
        assert "overfitting" in warning_text
