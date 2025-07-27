#!/usr/bin/env python3
"""
Demo script for the SimplePredictionAnalyzer with automatic mask inference.

This script demonstrates how to use the analyzer with automatic mask path
inference, so you only need to provide the image path and the mask directory.
"""

import logging
from pathlib import Path

from crackseg.evaluation.simple_prediction_analyzer import (
    SimplePredictionAnalyzer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_automatic_mask_inference():
    """Demonstrate automatic mask inference functionality."""

    # Configuration
    checkpoint_path = "outputs/checkpoints/model_best.pth.tar"
    config_path = "outputs/configurations/default_experiment/config.yaml"
    mask_dir = "data/train/masks"  # Directory containing ground truth masks

    # Test images from different directories
    test_images = [
        "data/train/images/98.jpg",  # Should find data/train/masks/98.png
        "data/val/images/1.jpg",  # Should find data/val/masks/1.png
        "data/test/images/1.jpg",  # Should find data/test/masks/1.png
    ]

    # Initialize analyzer with mask directory
    logger.info(
        "Initializing SimplePredictionAnalyzer with automatic mask "
        "inference..."
    )
    analyzer = SimplePredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mask_dir=mask_dir,
    )

    # Print model information
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
            output_path = f"outputs/predictions/{image_path.stem}_analysis.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            analyzer.create_visualization(result, output_path)
            logger.info(f"Visualization saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")


def demo_manual_mask_specification():
    """Demonstrate manual mask specification (for comparison)."""

    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Manual Mask Specification")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "outputs/checkpoints/model_best.pth.tar"
    config_path = "outputs/configurations/default_experiment/config.yaml"

    # Specific image and mask paths
    image_path = "data/train/images/98.jpg"
    mask_path = "data/train/masks/98.png"

    # Initialize analyzer without mask directory
    analyzer = SimplePredictionAnalyzer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    logger.info("Analyzing with manual mask specification:")
    logger.info(f"  Image: {image_path}")
    logger.info(f"  Mask: {mask_path}")

    try:
        result = analyzer.analyze_image(
            image_path=image_path,
            mask_path=mask_path,  # Explicitly specify mask
            threshold=0.5,
            auto_find_mask=False,  # Disable automatic inference
        )

        # Print results
        if "metrics" in result:
            logger.info("Analysis Results:")
            if "iou" in result:
                logger.info(f"  IoU: {result['iou']:.3f}")
            for metric, value in result["metrics"].items():
                logger.info(f"  {metric}: {value:.3f}")

        # Create visualization
        output_path = "outputs/predictions/manual_mask_analysis.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        analyzer.create_visualization(result, output_path)
        logger.info(f"Visualization saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")


def demo_prediction_only():
    """Demonstrate prediction-only analysis (no ground truth)."""

    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Prediction Only (No Ground Truth)")
    logger.info("=" * 60)

    # Configuration
    checkpoint_path = "outputs/checkpoints/model_best.pth.tar"
    config_path = "outputs/configurations/default_experiment/config.yaml"

    # Test image (no corresponding mask)
    image_path = "data/train/images/98.jpg"

    # Initialize analyzer without mask directory
    analyzer = SimplePredictionAnalyzer(
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
        output_path = "outputs/predictions/prediction_only_analysis.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        analyzer.create_visualization(result, output_path)
        logger.info(f"Visualization saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in prediction-only analysis: {e}")


def main():
    """Run all demonstration scenarios."""

    logger.info("SimplePredictionAnalyzer Demo with Automatic Mask Inference")
    logger.info("=" * 80)

    # Check if checkpoint exists
    checkpoint_path = Path("outputs/checkpoints/model_best.pth.tar")
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run a training experiment first.")
        return

    # Run demonstrations
    try:
        demo_automatic_mask_inference()
        demo_manual_mask_specification()
        demo_prediction_only()

        logger.info("\n" + "=" * 80)
        logger.info("All demonstrations completed successfully!")
        logger.info(
            "Check the 'outputs/predictions/' directory for visualizations."
        )

    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
