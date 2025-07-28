"""Command-line interface for crack segmentation prediction analysis."""

import argparse
import logging
import sys

from ..core.analyzer import PredictionAnalyzer
from ..visualization.prediction_viz import PredictionVisualizer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_model_info(analyzer: PredictionAnalyzer) -> None:
    """Print model information."""
    model_info = analyzer.get_model_info()
    print("Model Information:")
    print("-" * 40)
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()


def analyze_single_image(args: argparse.Namespace) -> None:
    """Analyze a single image."""
    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        mask_dir=args.mask_dir,
    )

    # Print model info
    print_model_info(analyzer)

    # Analyze image
    result = analyzer.analyze_image(
        image_path=args.image,
        mask_path=args.mask,
        threshold=args.threshold,
        auto_find_mask=not args.no_auto_mask,
    )

    # Print results
    if "metrics" in result:
        print("Analysis Results:")
        print("-" * 40)
        if "iou" in result:
            print(f"  IoU: {result['iou']:.3f}")
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:.3f}")
    else:
        print("Prediction completed (no ground truth available)")

    # Create visualization
    if args.output:
        visualizer = PredictionVisualizer(analyzer.config)
        visualizer.create_visualization(result, args.output)
        print(f"\nVisualization saved to: {args.output}")


def analyze_batch(args: argparse.Namespace) -> None:
    """Analyze a batch of images."""
    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        mask_dir=args.mask_dir,
    )

    # Print model info
    print_model_info(analyzer)

    # Process batch
    from ..metrics.batch_processor import BatchProcessor

    batch_processor = BatchProcessor(analyzer.image_processor)

    summary = batch_processor.process_batch(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        save_visualizations=not args.no_viz,
    )

    # Print summary
    print("Batch Analysis Summary:")
    print("-" * 40)
    for key, value in summary.items():
        if key != "all_metrics":
            print(f"  {key}: {value}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CrackSeg Prediction Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image with automatic mask inference
  python -m crackseg.evaluation.cli \\
      --checkpoint outputs/checkpoints/model_best.pth.tar \\
      --image data/test/images/98.jpg \\
      --mask-dir data/test/masks \\
      --output results/analysis.png

  # Analyze single image with manual mask
  python -m crackseg.evaluation.cli \\
      --checkpoint outputs/checkpoints/model_best.pth.tar \\
      --image data/test/images/98.jpg \\
      --mask data/test/masks/98.png \\
      --output results/analysis.png

  # Batch analysis
  python -m crackseg.evaluation.cli \\
      --checkpoint outputs/checkpoints/model_best.pth.tar \\
      --image-dir data/test/images \\
      --mask-dir data/test/masks \\
      --output-dir results/batch_analysis

  # Prediction only (no ground truth)
  python -m crackseg.evaluation.cli \\
      --checkpoint outputs/checkpoints/model_best.pth.tar \\
      --image data/test/images/98.jpg \\
      --no-auto-mask \\
      --output results/prediction.png
        """,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        help="Path to model configuration (optional)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use for inference (auto-detect if not specified)",
    )

    # Analysis mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--image",
        help="Path to single image for analysis",
    )
    mode_group.add_argument(
        "--image-dir",
        help="Directory of images for batch analysis",
    )

    # Ground truth arguments
    parser.add_argument(
        "--mask",
        help="Path to ground truth mask (for single image)",
    )
    parser.add_argument(
        "--mask-dir",
        help="Directory containing ground truth masks (for auto-inference)",
    )
    parser.add_argument(
        "--no-auto-mask",
        action="store_true",
        help="Disable automatic mask inference",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        help="Output path for single image visualization",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch analysis results",
    )

    # Analysis parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation (batch mode only)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Determine analysis mode
        if args.image:
            analyze_single_image(args)
        elif args.image_dir:
            analyze_batch(args)
        else:
            parser.error("Must specify either --image or --image-dir")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
