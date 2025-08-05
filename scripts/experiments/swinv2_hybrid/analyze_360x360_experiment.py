#!/usr/bin/env python3
"""
SwinV2 Hybrid 360x360 Experiment Analyzer

This script provides comprehensive analysis and monitoring for the
SwinV2 hybrid experiment with 360x360 images.

Features:
- Memory usage analysis and optimization recommendations
- Training progress monitoring
- Performance benchmarking
- Hardware utilization analysis
- Experiment comparison tools

Author: CrackSeg Project
Date: 2024
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.utils.memory import get_memory_stats  # noqa: E402


class ExperimentAnalyzer:
    """Analyzer for SwinV2 hybrid 360x360 experiments."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.config_file = self.experiment_dir / "config.yaml"
        self.results_file = self.experiment_dir / "final_results.yaml"
        self.log_file = self.experiment_dir / "training.log"

        # Load configuration
        if self.config_file.exists():
            self.config = OmegaConf.load(self.config_file)
        else:
            self.config = None
            logging.warning(f"Config file not found: {self.config_file}")

        # Load results
        if self.results_file.exists():
            self.results = OmegaConf.load(self.results_file)
        else:
            self.results = None
            logging.warning(f"Results file not found: {self.results_file}")

    def analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze memory usage patterns and provide recommendations."""
        logging.info("Analyzing memory usage...")

        memory_analysis = {
            "current_usage": {},
            "recommendations": [],
            "optimizations": [],
        }

        if torch.cuda.is_available():
            # Get current memory stats
            memory_stats = get_memory_stats()
            memory_analysis["current_usage"] = memory_stats

            # Analyze memory efficiency
            total_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
            allocated_memory = memory_stats["allocated_gb"]
            utilization = (allocated_memory / total_memory) * 100

            memory_analysis["utilization_percent"] = utilization

            # Generate recommendations
            if utilization > 90:
                memory_analysis["recommendations"].append(
                    "High memory utilization (>90%). Consider reducing batch size or enabling gradient checkpointing."
                )
            elif utilization < 50:
                memory_analysis["recommendations"].append(
                    "Low memory utilization (<50%). Consider increasing batch size for better efficiency."
                )

            # Check for memory leaks
            if memory_stats.get("peak_allocated_gb", 0) > total_memory * 0.95:
                memory_analysis["recommendations"].append(
                    "Memory usage peaked near capacity. Monitor for potential memory leaks."
                )

            # Optimization suggestions
            if self.config:
                batch_size = self.config.training.batch_size
                image_size = self.config.data.image_size[0]

                # Calculate theoretical memory usage
                theoretical_memory = self._calculate_theoretical_memory(
                    batch_size, image_size
                )

                if theoretical_memory > total_memory * 0.8:
                    memory_analysis["optimizations"].append(
                        f"Consider reducing batch size from {batch_size} to {max(1, batch_size - 1)}"
                    )

                if image_size > 256:
                    memory_analysis["optimizations"].append(
                        f"Large image size ({image_size}x{image_size}) may benefit from gradient accumulation"
                    )

        return memory_analysis

    def _calculate_theoretical_memory(
        self, batch_size: int, image_size: int
    ) -> float:
        """Calculate theoretical memory usage for the model."""
        # Rough estimation based on model parameters and image size
        # This is a simplified calculation
        model_params = 28e6  # SwinV2-tiny approximate parameters
        param_memory = model_params * 4 / 1024**3  # 4 bytes per parameter

        # Feature map memory (rough estimation)
        feature_memory = (
            batch_size * 3 * image_size * image_size * 4
        ) / 1024**3

        # Gradient memory
        gradient_memory = param_memory * 2  # Gradients + optimizer states

        return param_memory + feature_memory + gradient_memory

    def analyze_training_progress(self) -> dict[str, Any]:
        """Analyze training progress from log files."""
        logging.info("Analyzing training progress...")

        progress_analysis = {
            "epochs_completed": 0,
            "best_metrics": {},
            "convergence_analysis": {},
            "training_time": None,
        }

        if self.log_file.exists():
            # Parse log file for metrics
            metrics_data = self._parse_training_log()

            if metrics_data:
                progress_analysis["epochs_completed"] = len(
                    metrics_data.get("epochs", [])
                )
                progress_analysis["best_metrics"] = metrics_data.get(
                    "best_metrics", {}
                )
                progress_analysis["convergence_analysis"] = (
                    self._analyze_convergence(metrics_data)
                )
                progress_analysis["training_time"] = metrics_data.get(
                    "total_time"
                )

        return progress_analysis

    def _parse_training_log(self) -> dict[str, Any]:
        """Parse training log file for metrics."""
        metrics_data = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "best_metrics": {},
            "total_time": None,
        }

        try:
            with open(self.log_file) as f:
                lines = f.readlines()

            for line in lines:
                # Parse epoch information
                if "Epoch" in line and "val_iou" in line:
                    # Extract metrics from log line
                    # This is a simplified parser - adjust based on actual log format
                    pass

                # Parse timing information
                if "Training completed" in line:
                    # Extract total training time
                    pass

        except Exception as e:
            logging.warning(f"Could not parse log file: {e}")

        return metrics_data

    def _analyze_convergence(
        self, metrics_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze training convergence patterns."""
        convergence = {
            "converged": False,
            "plateau_detected": False,
            "overfitting_detected": False,
            "recommendations": [],
        }

        # Analyze loss trends
        train_loss = metrics_data.get("train_loss", [])
        val_loss = metrics_data.get("val_loss", [])

        if len(train_loss) > 10 and len(val_loss) > 10:
            # Check for convergence
            recent_train = train_loss[-10:]
            recent_val = val_loss[-10:]

            # Check if loss is still decreasing
            train_trend = np.mean(np.diff(recent_train))
            val_trend = np.mean(np.diff(recent_val))

            if abs(train_trend) < 0.001 and abs(val_trend) < 0.001:
                convergence["converged"] = True
                convergence["recommendations"].append(
                    "Training appears to have converged"
                )

            # Check for overfitting
            if val_trend > 0.001 and train_trend < -0.001:
                convergence["overfitting_detected"] = True
                convergence["recommendations"].append(
                    "Potential overfitting detected - consider early stopping"
                )

        return convergence

    def compare_with_baselines(self) -> dict[str, Any]:
        """Compare experiment results with baseline models."""
        logging.info("Comparing with baseline models...")

        if not self.results:
            return {"error": "No results available for comparison"}

        # Define baseline results for crack segmentation
        baselines = {
            "U-Net": {"iou": 0.681, "f1": 0.811, "dice": 0.792},
            "DeepLabV3+": {"iou": 0.688, "f1": 0.816, "dice": 0.799},
            "DeepCrack": {"iou": 0.869, "f1": 0.930, "dice": 0.918},
            "SwinV2-Base": {"iou": 0.852, "f1": 0.915, "dice": 0.903},
        }

        comparison = {
            "baselines": baselines,
            "our_results": dict(self.results),
            "improvements": {},
            "ranking": [],
        }

        # Convert DictConfig to dict if needed
        results_dict = dict(self.results) if self.results else {}

        # Calculate improvements
        for baseline_name, baseline_metrics in baselines.items():
            improvements = {}
            for metric in ["iou", "f1", "dice"]:
                if metric in results_dict and metric in baseline_metrics:
                    improvement = (
                        results_dict[metric] - baseline_metrics[metric]
                    )
                    improvements[metric] = improvement

            comparison["improvements"][baseline_name] = improvements

        # Create ranking
        all_results = []
        for name, metrics in baselines.items():
            all_results.append((name, metrics["iou"]))

        # Add our results
        if "iou" in results_dict:
            all_results.append(("Our Model", results_dict["iou"]))

        # Sort by IoU
        all_results.sort(key=lambda x: x[1], reverse=True)
        comparison["ranking"] = all_results

        return comparison

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        logging.info("Generating analysis report...")

        report = []
        report.append("=" * 80)
        report.append("SwinV2 Hybrid 360x360 Experiment Analysis Report")
        report.append("=" * 80)
        report.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append(f"Experiment Directory: {self.experiment_dir}")
        report.append("")

        # Memory analysis
        memory_analysis = self.analyze_memory_usage()
        report.append("MEMORY ANALYSIS")
        report.append("-" * 40)
        if memory_analysis["current_usage"]:
            report.append(
                f"Current Memory Usage: {memory_analysis['current_usage']}"
            )
        if memory_analysis["recommendations"]:
            report.append("Recommendations:")
            for rec in memory_analysis["recommendations"]:
                report.append(f"  - {rec}")
        report.append("")

        # Training progress
        progress_analysis = self.analyze_training_progress()
        report.append("TRAINING PROGRESS")
        report.append("-" * 40)
        report.append(
            f"Epochs Completed: {progress_analysis['epochs_completed']}"
        )
        if progress_analysis["best_metrics"]:
            report.append("Best Metrics:")
            for metric, value in progress_analysis["best_metrics"].items():
                report.append(f"  {metric}: {value:.4f}")
        report.append("")

        # Baseline comparison
        comparison = self.compare_with_baselines()
        report.append("BASELINE COMPARISON")
        report.append("-" * 40)
        if "ranking" in comparison:
            report.append("Model Ranking (by IoU):")
            for i, (name, iou) in enumerate(comparison["ranking"], 1):
                report.append(f"  {i}. {name}: {iou:.3f}")
        report.append("")

        # Hardware analysis
        if torch.cuda.is_available():
            report.append("HARDWARE ANALYSIS")
            report.append("-" * 40)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
            report.append(f"GPU: {gpu_name}")
            report.append(f"VRAM: {gpu_memory:.1f}GB")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        # Add recommendations based on analysis
        if memory_analysis.get("utilization_percent", 0) > 90:
            report.append(
                "- Consider reducing batch size or enabling gradient checkpointing"
            )

        if progress_analysis.get("convergence_analysis", {}).get(
            "overfitting_detected"
        ):
            report.append("- Implement early stopping to prevent overfitting")

        if self.config and self.config.training.batch_size < 4:
            report.append("- Consider increasing batch size if memory allows")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_analysis(self, output_file: Path | None = None) -> None:
        """Save analysis results to file."""
        if output_file is None:
            output_file = self.experiment_dir / "analysis_report.txt"

        report = self.generate_report()

        with open(output_file, "w") as f:
            f.write(report)

        logging.info(f"Analysis report saved to: {output_file}")


def main():
    """Main entry point for experiment analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze SwinV2 hybrid 360x360 experiment"
    )
    parser.add_argument(
        "experiment_dir", type=str, help="Path to experiment directory"
    )
    parser.add_argument(
        "--output", type=str, help="Output file for analysis report"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create analyzer
    analyzer = ExperimentAnalyzer(args.experiment_dir)

    # Generate and save analysis
    output_file = Path(args.output) if args.output else None
    analyzer.save_analysis(output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Memory summary
    memory_analysis = analyzer.analyze_memory_usage()
    if memory_analysis["current_usage"]:
        print(f"Memory Usage: {memory_analysis['current_usage']}")

    # Progress summary
    progress_analysis = analyzer.analyze_training_progress()
    print(
        f"Training Progress: {progress_analysis['epochs_completed']} epochs completed"
    )

    # Results summary
    if analyzer.results:
        print("Final Results:")
        for metric, value in analyzer.results.items():
            print(f"  {metric}: {value:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
