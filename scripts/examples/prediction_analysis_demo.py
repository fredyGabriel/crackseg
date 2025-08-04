#!/usr/bin/env python3
"""
Demo script for the new modular PredictionAnalyzer.

This script demonstrates the new modular architecture for crack segmentation
prediction analysis with automatic mask inference and professional
visualizations.
"""

import logging
from pathlib import Path

from crackseg.evaluation import PredictionAnalyzer, PredictionVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_automatic_mask_inference():
    """Demonstrate automatic mask inference functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Automatic Mask Inference")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "artifacts/checkpoints/model_best.pth.tar"
    config_path = "artifacts/configurations/default_experiment/config.yaml"
    mask_dir = "data/unified/masks"

    # Test images
    test_images = [
        "data/unified/images/98.jpg",
        "data/unified/images/99.jpg",
        "data/unified/images/100.jpg",
    ]

    # Initialize analyzer with mask directory for auto-inference
    logger.info(
        "Initializing PredictionAnalyzer with automatic mask inference"
    )
    analyzer = PredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mask_dir=mask_dir,  # Enable auto-inference
    )

    # Print model info
    model_info = analyzer.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Test automatic mask inference
    for image_path in test_images:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Analyzing image: {image_path}")

        try:
            # This will automatically find the corresponding mask
            result = analyzer.analyze_image(
                image_path=image_path,
                threshold=0.5,
                auto_find_mask=True,  # Enable automatic mask inference
            )

            # Print results
            if "metrics" in result:
                logger.info("Analysis Results:")
                if "iou" in result:
                    logger.info(f"  IoU: {result['iou']:.3f}")
                for metric, value in result["metrics"].items():
                    logger.info(f"  {metric}: {value:.3f}")
            else:
                logger.info("Prediction completed (no ground truth found)")

            # Create visualization
            output_path = (
                f"artifacts/predictions/{image_path.stem}_analysis.png"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            visualizer = PredictionVisualizer(analyzer.config)
            visualizer.create_visualization(result, output_path)
            logger.info(f"Visualization saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")


def demo_prediction_only():
    """Demonstrate prediction-only analysis (no ground truth)."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Prediction Only (No Ground Truth)")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "artifacts/checkpoints/model_best.pth.tar"
    config_path = "artifacts/configurations/default_experiment/config.yaml"

    # Test image (no corresponding mask)
    image_path = "data/unified/images/98.jpg"

    # Initialize analyzer without mask directory
    analyzer = PredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    logger.info("Performing prediction-only analysis:")
    logger.info(f"  Image: {image_path}")

    try:
        result = analyzer.analyze_image(
            image_path=image_path,
            threshold=0.5,
            auto_find_mask=False,  # Disable automatic inference
        )

        logger.info("Prediction completed successfully")
        logger.info(f"  Prediction shape: {result['prediction_shape']}")
        logger.info(f"  Threshold used: {result['threshold']}")

        # Create visualization
        output_path = "artifacts/predictions/prediction_only_analysis.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        visualizer = PredictionVisualizer(analyzer.config)
        visualizer.create_visualization(result, output_path)
        logger.info(f"Visualization saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in prediction-only analysis: {e}")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Batch Processing")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "artifacts/checkpoints/model_best.pth.tar"
    config_path = "artifacts/configurations/default_experiment/config.yaml"
    image_dir = "data/unified/images"
    mask_dir = "data/unified/masks"

    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mask_dir=mask_dir,
    )

    # Import batch processor
    from crackseg.evaluation import BatchProcessor

    batch_processor = BatchProcessor(analyzer.image_processor)

    logger.info("Performing batch analysis:")
    logger.info(f"  Image directory: {image_dir}")
    logger.info(f"  Mask directory: {mask_dir}")

    try:
        summary = batch_processor.process_batch(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir="artifacts/batch_analysis",
            save_visualizations=True,
        )

        logger.info("Batch analysis completed successfully")
        logger.info("Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")


def demo_visualization_features():
    """Demonstrate advanced visualization features."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Advanced Visualization Features")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "artifacts/checkpoints/model_best.pth.tar"
    config_path = "artifacts/configurations/default_experiment/config.yaml"
    mask_dir = "data/unified/masks"

    # Initialize analyzer and visualizer
    analyzer = PredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mask_dir=mask_dir,
    )
    visualizer = PredictionVisualizer(analyzer.config)

    # Test images for comparison
    test_images = [
        "data/unified/images/98.jpg",
        "data/unified/images/99.jpg",
        "data/unified/images/100.jpg",
    ]

    # Analyze multiple images
    results = []
    for image_path in test_images:
        image_path = Path(image_path)
        if not image_path.exists():
            continue

        try:
            result = analyzer.analyze_image(
                image_path=image_path,
                threshold=0.5,
                auto_find_mask=True,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")

    if results:
        # Create comparison grid
        output_path = "artifacts/predictions/comparison_grid.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        visualizer.create_comparison_grid(
            results, save_path=output_path, max_images=9
        )
        logger.info(f"Comparison grid saved to: {output_path}")


def main():
    """Run all demos."""
    logger.info("CrackSeg Prediction Analysis Demo")
    logger.info("New Modular Architecture")
    logger.info("=" * 60)

    # Run demos
    demo_automatic_mask_inference()
    demo_prediction_only()
    demo_batch_processing()
    demo_visualization_features()

    logger.info("\n" + "=" * 60)
    logger.info("All demos completed successfully!")
    logger.info("Check artifacts/predictions/ for generated visualizations")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
