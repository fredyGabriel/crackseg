"""Batch processing utilities for crack segmentation evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..core.image_processor import ImageProcessor
from .calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple images for batch evaluation."""

    def __init__(self, image_processor: ImageProcessor) -> None:
        """
        Initialize the batch processor.

        Args:
            image_processor: Image processor instance
        """
        self.image_processor = image_processor
        self.metrics_calculator = MetricsCalculator()

    def process_batch(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        save_visualizations: bool = True,
    ) -> dict[str, Any]:
        """
        Process a batch of images for evaluation.

        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing ground truth masks (optional)
            output_dir: Directory to save results (optional)
            save_visualizations: Whether to save individual visualizations

        Returns:
            Dictionary containing batch analysis results
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        output_dir = Path(output_dir) if output_dir else None

        # Find image files
        image_files = self.image_processor.find_image_files(image_dir)
        logger.info(f"Found {len(image_files)} images for batch analysis")

        # Process each image
        results = []
        all_metrics = []

        for i, image_file in enumerate(image_files):
            logger.info(
                f"Processing {i + 1}/{len(image_files)}: {image_file.name}"
            )

            try:
                # Find corresponding mask if available
                mask_file = self.image_processor.find_corresponding_mask(
                    image_file, mask_dir
                )

                # Store result info
                result_info = {
                    "image_path": str(image_file),
                    "mask_path": str(mask_file) if mask_file else None,
                    "has_ground_truth": mask_file is not None,
                }

                results.append(result_info)

                if mask_file:
                    all_metrics.append(
                        {
                            "image": image_file.name,
                            "mask": mask_file.name,
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue

        # Calculate aggregate metrics
        batch_summary = {
            "total_images": len(image_files),
            "processed_images": len(results),
            "failed_images": len(image_files) - len(results),
            "images_with_ground_truth": len(all_metrics),
        }

        # Save batch results
        if output_dir:
            self._save_batch_results(output_dir, batch_summary, results)

        return batch_summary

    def _save_batch_results(
        self,
        output_dir: Path,
        batch_summary: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        """
        Save batch processing results to files.

        Args:
            output_dir: Directory to save results
            batch_summary: Summary of batch processing
            results: Detailed results for each image
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "batch_analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

        # Save detailed results
        results_path = output_dir / "detailed_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Batch results saved to: {output_dir}")

    def calculate_average_metrics(
        self, metrics_list: list[dict[str, float]]
    ) -> dict[str, float]:
        """
        Calculate average metrics from a list of metric dictionaries.

        Args:
            metrics_list: List of metric dictionaries

        Returns:
            Dictionary containing average metrics
        """
        if not metrics_list:
            return {}

        # Get all metric keys
        metric_keys = metrics_list[0].keys()
        avg_metrics = {}

        for key in metric_keys:
            values = [
                metrics[key] for metrics in metrics_list if key in metrics
            ]
            if values:
                avg_metrics[key] = float(np.mean(values))

        return avg_metrics

    def generate_batch_report(
        self, batch_summary: dict[str, Any], output_dir: Path
    ) -> None:
        """
        Generate a comprehensive batch processing report.

        Args:
            batch_summary: Summary of batch processing
            output_dir: Directory to save the report
        """
        report_path = output_dir / "batch_report.txt"

        with open(report_path, "w") as f:
            f.write("CrackSeg Batch Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            f.write("Processing Summary:\n")
            f.write(f"  Total images: {batch_summary['total_images']}\n")
            f.write(
                f"  Successfully processed: "
                f"{batch_summary['processed_images']}\n"
            )
            f.write(f"  Failed: {batch_summary['failed_images']}\n")
            f.write(
                f"  With ground truth: "
                f"{batch_summary['images_with_ground_truth']}\n\n"
            )

            if batch_summary["failed_images"] > 0:
                f.write(
                    "⚠️  Some images failed to process. "
                    "Check logs for details.\n\n"
                )

            f.write("Files generated:\n")
            f.write("  - batch_analysis_summary.json: Processing summary\n")
            f.write("  - detailed_results.json: Per-image results\n")
            f.write("  - batch_report.txt: This report\n")

        logger.info(f"Batch report saved to: {report_path}")
