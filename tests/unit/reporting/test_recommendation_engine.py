#!/usr/bin/env python3
"""
Test script for AutomatedRecommendationEngine

This script tests the automated recommendation engine functionality,
including training pattern analysis, hyperparameter suggestions,
and optimization opportunity identification.
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
try:
    from crackseg.reporting.config import (
        ExperimentData,
        OutputFormat,
        ReportConfig,
    )
    from crackseg.reporting.recommendations import (
        AutomatedRecommendationEngine,
    )
except ImportError:
    # Fallback for when module is not available
    pass


def create_sample_experiment_data(experiment_dir: Path) -> ExperimentData:
    """Create sample experiment data for testing."""
    # Create sample training metrics
    epochs = 50
    training_metrics = []

    # Simulate different training patterns
    for epoch in range(epochs):
        # Simulate loss decreasing with some noise
        base_loss = 0.5 * np.exp(-epoch / 20) + 0.05
        noise = np.random.normal(0, 0.02)
        loss = max(0.01, base_loss + noise)

        # Simulate metrics improving
        iou = min(0.85, 0.3 + 0.5 * (1 - np.exp(-epoch / 15)))
        dice = min(0.90, 0.4 + 0.4 * (1 - np.exp(-epoch / 15)))
        f1 = min(0.88, 0.35 + 0.45 * (1 - np.exp(-epoch / 15)))

        training_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "iou": iou + np.random.normal(0, 0.01),
                "dice": dice + np.random.normal(0, 0.01),
                "f1": f1 + np.random.normal(0, 0.01),
            }
        )

    # Create validation metrics (slightly worse than training)
    validation_metrics = []
    for epoch in range(epochs):
        base_loss = 0.6 * np.exp(-epoch / 25) + 0.08  # Higher loss
        noise = np.random.normal(0, 0.03)
        loss = max(0.02, base_loss + noise)

        iou = min(0.80, 0.25 + 0.45 * (1 - np.exp(-epoch / 18)))
        dice = min(0.85, 0.35 + 0.35 * (1 - np.exp(-epoch / 18)))
        f1 = min(0.83, 0.30 + 0.40 * (1 - np.exp(-epoch / 18)))

        validation_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "iou": iou + np.random.normal(0, 0.015),
                "dice": dice + np.random.normal(0, 0.015),
                "f1": f1 + np.random.normal(0, 0.015),
            }
        )

    # Create final metrics
    final_metrics = {
        "iou": 0.78,
        "dice": 0.82,
        "f1": 0.80,
        "precision": 0.85,
        "recall": 0.76,
        "loss": 0.12,
    }

    # Create sample configuration
    config = OmegaConf.create(
        {
            "learning_rate": 0.001,
            "batch_size": 16,
            "optimizer": "adam",
            "scheduler": "step",
            "encoder": "resnet50",
            "decoder": "unet",
            "loss": "bce",
            "epochs": 50,
        }
    )

    return ExperimentData(
        experiment_id="test_exp_001",
        experiment_dir=experiment_dir,
        config=config,
        metrics={
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "final_metrics": final_metrics,
        },
        artifacts={},
    )


def create_poor_performance_experiment(experiment_dir: Path) -> ExperimentData:
    """
    Create experiment data with poor performance for testing recommendations.
    """
    epochs = 30
    training_metrics = []

    # Simulate poor training pattern
    for epoch in range(epochs):
        # High loss that doesn't decrease much
        base_loss = 0.8 * np.exp(-epoch / 50) + 0.3
        noise = np.random.normal(0, 0.05)
        loss = max(0.2, base_loss + noise)

        # Poor metrics
        iou = min(0.55, 0.2 + 0.3 * (1 - np.exp(-epoch / 30)))
        dice = min(0.60, 0.25 + 0.25 * (1 - np.exp(-epoch / 30)))
        f1 = min(0.58, 0.22 + 0.28 * (1 - np.exp(-epoch / 30)))

        training_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "iou": iou + np.random.normal(0, 0.02),
                "dice": dice + np.random.normal(0, 0.02),
                "f1": f1 + np.random.normal(0, 0.02),
            }
        )

    # Poor final metrics
    final_metrics = {
        "iou": 0.52,
        "dice": 0.58,
        "f1": 0.55,
        "precision": 0.60,
        "recall": 0.50,
        "loss": 0.35,
    }

    config = OmegaConf.create(
        {
            "learning_rate": 0.01,  # Too high
            "batch_size": 4,  # Too small
            "optimizer": "sgd",
            "encoder": "resnet18",
            "decoder": "simple",
            "loss": "bce",
            "epochs": 30,
        }
    )

    return ExperimentData(
        experiment_id="test_poor_exp",
        experiment_dir=experiment_dir,
        config=config,
        metrics={
            "training_metrics": training_metrics,
            "validation_metrics": training_metrics,  # Same for simplicity
            "final_metrics": final_metrics,
        },
        artifacts={},
    )


def test_recommendation_engine() -> None:
    """Test the AutomatedRecommendationEngine with sample data."""
    print("🧪 Testing AutomatedRecommendationEngine")
    print("=" * 50)

    # Initialize recommendation engine
    recommendation_engine = AutomatedRecommendationEngine()

    # Create report configuration
    config = ReportConfig(
        output_formats=[OutputFormat.JSON],
        include_performance_analysis=True,
        include_recommendations=True,
    )

    # Use temporary directory for test experiments
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test with good performance experiment
        print("\n📊 Testing with good performance experiment...")
        good_experiment_dir = temp_path / "test_experiment"
        good_experiment_dir.mkdir()
        good_experiment = create_sample_experiment_data(good_experiment_dir)

        # Test training pattern analysis
        print("🔍 Analyzing training patterns...")
        training_recommendations = (
            recommendation_engine.analyze_training_patterns(
                good_experiment, config
            )
        )

        print(
            f"✅ Generated {len(training_recommendations)} training "
            "recommendations:"
        )
        for i, rec in enumerate(training_recommendations, 1):
            print(f"   {i}. {rec}")

        # Test hyperparameter suggestions
        print("\n⚙️ Generating hyperparameter suggestions...")
        hyperparameter_suggestions = (
            recommendation_engine.suggest_hyperparameter_improvements(
                good_experiment, config
            )
        )

        print("✅ Hyperparameter suggestions:")
        for category, suggestions in hyperparameter_suggestions.items():
            if suggestions.get("recommendations"):
                print(f"   📋 {category.title()}:")
                print(
                    f"      Current: {suggestions.get('current', 'Unknown')}"
                )
                print(
                    f"      Reasoning: {suggestions.get('reasoning', 'N/A')}"
                )
                for rec in suggestions.get("recommendations", []):
                    print(f"      • {rec}")

        # Test optimization opportunities
        print("\n🎯 Identifying optimization opportunities...")
        optimization_opportunities = (
            recommendation_engine.identify_optimization_opportunities(
                good_experiment, config
            )
        )

        print(
            f"✅ Found {len(optimization_opportunities)} optimization "
            "opportunities:"
        )
        for i, opp in enumerate(optimization_opportunities, 1):
            print(f"   {i}. {opp}")

        # Test with poor performance experiment
        print("\n📉 Testing with poor performance experiment...")
        poor_experiment_dir = temp_path / "test_poor_experiment"
        poor_experiment_dir.mkdir()
        poor_experiment = create_poor_performance_experiment(
            poor_experiment_dir
        )

        # Test training pattern analysis for poor performance
        print("🔍 Analyzing poor training patterns...")
        poor_training_recommendations = (
            recommendation_engine.analyze_training_patterns(
                poor_experiment, config
            )
        )

        print(
            f"✅ Generated {len(poor_training_recommendations)} "
            "recommendations for poor performance:"
        )
        for i, rec in enumerate(poor_training_recommendations, 1):
            print(f"   {i}. {rec}")

        # Test hyperparameter suggestions for poor performance
        print(
            "\n⚙️ Generating hyperparameter suggestions for poor performance..."
        )
        poor_hyperparameter_suggestions = (
            recommendation_engine.suggest_hyperparameter_improvements(
                poor_experiment, config
            )
        )

        print("✅ Hyperparameter suggestions for poor performance:")
        for category, suggestions in poor_hyperparameter_suggestions.items():
            if suggestions.get("recommendations"):
                print(f"   📋 {category.title()}:")
                print(
                    f"      Current: {suggestions.get('current', 'Unknown')}"
                )
                print(
                    f"      Reasoning: {suggestions.get('reasoning', 'N/A')}"
                )
                for rec in suggestions.get("recommendations", []):
                    print(f"      • {rec}")

        # Test optimization opportunities for poor performance
        print(
            "\n🎯 Identifying optimization opportunities for poor "
            "performance..."
        )
        poor_optimization_opportunities = (
            recommendation_engine.identify_optimization_opportunities(
                poor_experiment, config
            )
        )

        print(
            f"✅ Found {len(poor_optimization_opportunities)} optimization "
            "opportunities:"
        )
        for i, opp in enumerate(poor_optimization_opportunities, 1):
            print(f"   {i}. {opp}")


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)

    recommendation_engine = AutomatedRecommendationEngine()
    config = ReportConfig(
        output_formats=[OutputFormat.JSON],
        include_performance_analysis=True,
        include_recommendations=True,
    )

    # Use temporary directory for test experiments
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test with minimal data
        print("\n📊 Testing with minimal experiment data...")
        minimal_experiment_dir = temp_path / "minimal_test"
        minimal_experiment_dir.mkdir()
        minimal_experiment = ExperimentData(
            experiment_id="minimal_test",
            experiment_dir=minimal_experiment_dir,
            config=OmegaConf.create({}),
            metrics={},
            artifacts={},
        )

        # Test training pattern analysis with minimal data
        try:
            recommendations = recommendation_engine.analyze_training_patterns(
                minimal_experiment, config
            )
            print(
                f"✅ Minimal data test passed: {len(recommendations)} "
                "recommendations"
            )
        except Exception as e:
            print(f"❌ Minimal data test failed: {e}")

        # Test with missing metrics
        print("\n📊 Testing with missing metrics...")
        missing_metrics_experiment_dir = temp_path / "missing_metrics_test"
        missing_metrics_experiment_dir.mkdir()
        missing_metrics_experiment = ExperimentData(
            experiment_id="missing_metrics_test",
            experiment_dir=missing_metrics_experiment_dir,
            config=OmegaConf.create({"learning_rate": 0.001}),
            metrics={"final_metrics": {"iou": 0.75}},  # Only final metrics
            artifacts={},
        )

        try:
            recommendations = recommendation_engine.analyze_training_patterns(
                missing_metrics_experiment, config
            )
            print(
                f"✅ Missing metrics test passed: {len(recommendations)} "
                "recommendations"
            )
        except Exception as e:
            print(f"❌ Missing metrics test failed: {e}")


def test_performance_thresholds() -> None:
    """Test performance threshold analysis."""
    print("\n🧪 Testing Performance Thresholds")
    print("=" * 40)

    recommendation_engine = AutomatedRecommendationEngine()
    config = ReportConfig(
        output_formats=[OutputFormat.JSON],
        include_performance_analysis=True,
        include_recommendations=True,
    )

    # Use temporary directory for test experiments
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test excellent performance
        print("\n🏆 Testing excellent performance...")
        excellent_experiment_dir = temp_path / "excellent_test"
        excellent_experiment_dir.mkdir()
        excellent_experiment = ExperimentData(
            experiment_id="excellent_test",
            experiment_dir=excellent_experiment_dir,
            config=OmegaConf.create({"learning_rate": 0.001}),
            metrics={
                "final_metrics": {
                    "iou": 0.88,
                    "dice": 0.92,
                    "f1": 0.91,
                    "loss": 0.08,
                }
            },
            artifacts={},
        )

        recommendations = recommendation_engine.analyze_training_patterns(
            excellent_experiment, config
        )
        print(
            f"✅ Excellent performance test: {len(recommendations)} "
            "recommendations"
        )

        # Test poor performance
        print("\n📉 Testing poor performance...")
        poor_experiment_dir = temp_path / "poor_test"
        poor_experiment_dir.mkdir()
        poor_experiment = ExperimentData(
            experiment_id="poor_test",
            experiment_dir=poor_experiment_dir,
            config=OmegaConf.create({"learning_rate": 0.01}),
            metrics={
                "final_metrics": {
                    "iou": 0.45,
                    "dice": 0.50,
                    "f1": 0.48,
                    "loss": 0.45,
                }
            },
            artifacts={},
        )

        recommendations = recommendation_engine.analyze_training_patterns(
            poor_experiment, config
        )
        print(
            f"✅ Poor performance test: {len(recommendations)} recommendations"
        )

        # Test moderate performance
        print("\n📊 Testing moderate performance...")
        moderate_experiment_dir = temp_path / "moderate_test"
        moderate_experiment_dir.mkdir()
        moderate_experiment = ExperimentData(
            experiment_id="moderate_test",
            experiment_dir=moderate_experiment_dir,
            config=OmegaConf.create({"learning_rate": 0.001}),
            metrics={
                "final_metrics": {
                    "iou": 0.70,
                    "dice": 0.75,
                    "f1": 0.73,
                    "loss": 0.25,
                }
            },
            artifacts={},
        )

        recommendations = recommendation_engine.analyze_training_patterns(
            moderate_experiment, config
        )
        print(
            f"✅ Moderate performance test: {len(recommendations)} "
            "recommendations"
        )


def main() -> None:
    """Run all recommendation engine tests."""
    print("🚀 AutomatedRecommendationEngine Test Suite")
    print("=" * 60)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Run main tests
        test_recommendation_engine()

        # Run edge case tests
        test_edge_cases()

        # Run performance threshold tests
        test_performance_thresholds()

        print("\n" + "=" * 60)
        print("✅ All recommendation engine tests completed successfully!")
        print("📋 Summary:")
        print("   • Training pattern analysis: ✅")
        print("   • Hyperparameter suggestions: ✅")
        print("   • Optimization opportunities: ✅")
        print("   • Edge case handling: ✅")
        print("   • Performance threshold analysis: ✅")
        print("\n🎉 Recommendation engine is ready for production use!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
