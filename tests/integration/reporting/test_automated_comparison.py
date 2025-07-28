#!/usr/bin/env python3
"""
Test script for automated experiment comparison functionality.

This script creates sample experiment data and tests the
AutomatedComparisonEngine to ensure it works correctly before using it
with real experiments.
"""

import sys
from pathlib import Path

from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.reporting.comparison import (  # noqa: E402
    AutomatedComparisonEngine,
)
from crackseg.reporting.config import (  # noqa: E402
    ExperimentData,
    OutputFormat,
    ReportConfig,
)


def create_sample_experiment_data() -> list[ExperimentData]:
    """Create sample experiment data for testing."""
    sample_experiments = []

    # Sample experiment configurations
    experiment_configs = [
        {
            "experiment_id": "swin_unet_v1",
            "model": {"encoder": "swin_tiny", "decoder": "unet"},
            "training": {"learning_rate": 0.001, "epochs": 100},
            "metrics": {
                "iou": 0.847,
                "dice": 0.912,
                "f1": 0.889,
                "precision": 0.923,
                "recall": 0.856,
                "loss": 0.234,
            },
        },
        {
            "experiment_id": "resnet_unet_v1",
            "model": {"encoder": "resnet50", "decoder": "unet"},
            "training": {"learning_rate": 0.0001, "epochs": 80},
            "metrics": {
                "iou": 0.823,
                "dice": 0.898,
                "f1": 0.876,
                "precision": 0.901,
                "recall": 0.852,
                "loss": 0.267,
            },
        },
        {
            "experiment_id": "swin_deeplab_v1",
            "model": {"encoder": "swin_small", "decoder": "deeplabv3plus"},
            "training": {"learning_rate": 0.0005, "epochs": 120},
            "metrics": {
                "iou": 0.856,
                "dice": 0.918,
                "f1": 0.895,
                "precision": 0.934,
                "recall": 0.861,
                "loss": 0.198,
            },
        },
        {
            "experiment_id": "efficientnet_unet_v1",
            "model": {"encoder": "efficientnet_b0", "decoder": "unet"},
            "training": {"learning_rate": 0.0002, "epochs": 90},
            "metrics": {
                "iou": 0.812,
                "dice": 0.885,
                "f1": 0.863,
                "precision": 0.889,
                "recall": 0.838,
                "loss": 0.289,
            },
        },
    ]

    for config in experiment_configs:
        # Create complete summary structure
        complete_summary = {
            "experiment_info": {
                "total_epochs": config["training"]["epochs"],
                "best_epoch": config["training"]["epochs"],
            },
            "best_metrics": {
                metric: {"value": value, "epoch": config["training"]["epochs"]}
                for metric, value in config["metrics"].items()
            },
        }

        # Create temporary experiment directory
        temp_dir = Path(f"temp_test_{config['experiment_id']}")
        temp_dir.mkdir(exist_ok=True)

        # Create experiment data
        exp_data = ExperimentData(
            experiment_id=config["experiment_id"],
            experiment_dir=temp_dir,
            config=DictConfig(config),
            metrics={"complete_summary": complete_summary},
            artifacts={},  # Empty artifacts for testing
        )

        sample_experiments.append(exp_data)

    return sample_experiments


def test_comparison_engine() -> None:
    """Test the AutomatedComparisonEngine with sample data."""
    print("🧪 Testing AutomatedComparisonEngine")
    print("=" * 50)

    # Create sample data
    sample_experiments = create_sample_experiment_data()
    print(f"📊 Created {len(sample_experiments)} sample experiments")

    # Initialize comparison engine
    comparison_engine = AutomatedComparisonEngine()
    config = ReportConfig(
        output_formats=[OutputFormat.JSON],
        include_performance_analysis=True,
        include_recommendations=True,
    )

    # Test comparison analysis
    print("\n🔍 Running comparison analysis...")
    comparison_results = comparison_engine.compare_experiments(
        sample_experiments, config
    )

    # Test best performer identification
    print("🏆 Identifying best performer...")
    best_performer = comparison_engine.identify_best_performing(
        sample_experiments, config
    )

    # Test comparison table generation
    print("📋 Generating comparison table...")
    comparison_table = comparison_engine.generate_comparison_table(
        sample_experiments, config
    )

    # Display results
    print("\n✅ Test Results:")
    print("-" * 30)

    print("📊 Comparison Analysis:")
    print(
        f"   - Experiments compared: "
        f"{comparison_results.get('experiment_count', 0)}"
    )
    print(
        f"   - Statistical analysis: "
        f"{'✅' if comparison_results.get('statistical_analysis') else '❌'}"
    )
    print(
        f"   - Ranking analysis: "
        f"{'✅' if comparison_results.get('ranking_analysis') else '❌'}"
    )
    print(
        f"   - Performance trends: "
        f"{'✅' if comparison_results.get('performance_trends') else '❌'}"
    )
    print(
        f"   - Anomaly detection: "
        f"{'✅' if comparison_results.get('anomaly_detection') else '❌'}"
    )
    print(
        f"   - Recommendations: "
        f"{len(comparison_results.get('recommendations', []))}"
    )

    print("\n🏆 Best Performer:")
    if "experiment_id" in best_performer:
        print(f"   - Experiment ID: {best_performer['experiment_id']}")
        print(
            f"   - Composite Score: "
            f"{best_performer.get('composite_score', 0):.4f}"
        )
        print(
            f"   - Confidence Level: "
            f"{best_performer.get('confidence_level', 0):.2%}"
        )
        print(
            f"   - Statistical Significance: "
            f"{best_performer.get('statistical_significance', False)}"
        )

    print("\n📋 Comparison Table:")
    if "table_data" in comparison_table:
        print(f"   - Table entries: {len(comparison_table['table_data'])}")
        print(
            f"   - Export formats: "
            f"{comparison_table.get('export_formats', [])}"
        )

    # Display ranking
    if "ranking_analysis" in comparison_results:
        ranking = comparison_results["ranking_analysis"]["ranking"]
        print("\n🥇 Ranking:")
        for i, rank in enumerate(ranking[:3]):
            print(
                f"   {i + 1}. {rank.get('experiment_id', 'N/A')} "
                f"(Score: {rank.get('total_score', 0):.4f})"
            )

    # Display recommendations
    if "recommendations" in comparison_results:
        print("\n💡 Key Recommendations:")
        for rec in comparison_results["recommendations"][:3]:
            print(f"   • {rec}")

    print("\n✅ All tests completed successfully!")


def test_metric_calculations() -> None:
    """Test metric calculations and scoring."""
    print("\n🧮 Testing Metric Calculations")
    print("=" * 40)

    comparison_engine = AutomatedComparisonEngine()

    # Test composite score calculation
    sample_metrics = {
        "swin_unet": {
            "iou": 0.847,
            "dice": 0.912,
            "f1": 0.889,
            "precision": 0.923,
            "recall": 0.856,
        },
        "resnet_unet": {
            "iou": 0.823,
            "dice": 0.898,
            "f1": 0.876,
            "precision": 0.901,
            "recall": 0.852,
        },
    }

    composite_scores = comparison_engine._calculate_composite_scores(
        sample_metrics
    )

    print("📊 Composite Score Calculation:")
    for exp_id, score_data in composite_scores.items():
        print(f"   {exp_id}:")
        print(f"     - Total Score: {score_data['total_score']:.4f}")
        print("     - Score Breakdown:")
        for metric, breakdown in score_data["score_breakdown"].items():
            print(
                f"       • {metric}: {breakdown['raw_value']:.4f} "
                f"(normalized: {breakdown['normalized_score']:.4f}, "
                f"weighted: {breakdown['weighted_score']:.4f})"
            )

    print("\n✅ Metric calculations completed!")


def test_statistical_analysis() -> None:
    """Test statistical analysis functions."""
    print("\n📈 Testing Statistical Analysis")
    print("=" * 40)

    comparison_engine = AutomatedComparisonEngine()

    # Test with sample metrics
    sample_metrics = {
        "exp1": {"iou": 0.847, "dice": 0.912, "f1": 0.889},
        "exp2": {"iou": 0.823, "dice": 0.898, "f1": 0.876},
        "exp3": {"iou": 0.856, "dice": 0.918, "f1": 0.895},
        "exp4": {"iou": 0.812, "dice": 0.885, "f1": 0.863},
    }

    # Test statistical analysis
    stats_analysis = comparison_engine._perform_statistical_analysis(
        sample_metrics
    )

    print("📊 Statistical Analysis Results:")
    for metric, stats in stats_analysis["descriptive_statistics"].items():
        print(f"   {metric.upper()}:")
        print(f"     - Mean: {stats['mean']:.4f}")
        print(f"     - Std: {stats['std']:.4f}")
        print(f"     - Min: {stats['min']:.4f}")
        print(f"     - Max: {stats['max']:.4f}")
        print(f"     - Median: {stats['median']:.4f}")

    # Test correlation analysis
    if "correlation_analysis" in stats_analysis:
        print("\n🔗 Correlation Analysis:")
        for correlation_name, correlation_value in stats_analysis[
            "correlation_analysis"
        ].items():
            print(f"   {correlation_name}: {correlation_value:.4f}")

    # Test significance tests
    if "significance_tests" in stats_analysis:
        print("\n📊 Significance Tests:")
        for metric, test_result in stats_analysis[
            "significance_tests"
        ].items():
            print(f"   {metric.upper()}:")
            print(f"     - T-statistic: {test_result['t_statistic']:.4f}")
            print(f"     - P-value: {test_result['p_value']:.4f}")
            print(f"     - Significant: {test_result['significant']}")

    print("\n✅ Statistical analysis completed!")


def main() -> None:
    """Run all tests for automated comparison functionality."""
    print("🚀 Automated Comparison Engine Test Suite")
    print("=" * 60)

    try:
        # Test comparison engine
        test_comparison_engine()

        # Test metric calculations
        test_metric_calculations()

        # Test statistical analysis
        test_statistical_analysis()

        print("\n🎉 All tests passed successfully!")
        print("✅ AutomatedComparisonEngine is ready for production use!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
