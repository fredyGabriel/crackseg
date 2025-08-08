#!/usr/bin/env python3
"""
SwinV2 Hybrid Experiment Analyzer - Generic Version

This script provides comprehensive analysis and monitoring for any
SwinV2 hybrid experiment regardless of dataset or image size.

Features:
- Memory usage analysis and optimization recommendations
- Training progress monitoring
- Performance benchmarking
- Hardware utilization analysis
- Experiment comparison tools
- Dataset-specific analysis
- Architecture-agnostic metrics

Author: CrackSeg Project
Date: 2024
"""

import argparse
import json
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

from crackseg.data.memory import get_gpu_memory_usage  # noqa: E402


class GenericExperimentAnalyzer:
    """Generic analyzer for SwinV2 hybrid experiments with any dataset/size."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

        # Try different possible config file locations
        config_locations = [
            self.experiment_dir / "config.yaml",
            self.experiment_dir / "configurations" / "training_config.yaml",
            self.experiment_dir
            / "configurations"
            / f"{self.experiment_dir.name}"
            / "training_config.yaml",
        ]

        self.config_file = None
        for loc in config_locations:
            if loc.exists():
                self.config_file = loc
                break

        # Try different possible results file locations
        results_locations = [
            self.experiment_dir / "final_results.yaml",
            self.experiment_dir / "metrics" / "complete_summary.json",
            self.experiment_dir / "metrics" / "summary.json",
        ]

        self.results_file = None
        for loc in results_locations:
            if loc.exists():
                self.results_file = loc
                break

        self.log_file = self.experiment_dir / "training.log"

        # Load configuration
        if self.config_file and self.config_file.exists():
            self.config = OmegaConf.load(self.config_file)
        else:
            self.config = None
            logging.warning(f"Config file not found: {self.config_file}")

        # Load results
        if self.results_file and self.results_file.exists():
            if self.results_file.suffix == ".json":
                with open(self.results_file) as f:
                    self.results = json.load(f)
            else:
                self.results = OmegaConf.load(self.results_file)
        else:
            self.results = None
            logging.warning(f"Results file not found: {self.results_file}")

        # Extract experiment metadata
        self.experiment_metadata = self._extract_experiment_metadata()

    def _extract_experiment_metadata(self) -> dict[str, Any]:
        """Extract key metadata from experiment configuration."""
        metadata = {
            "dataset": "unknown",
            "image_size": "unknown",
            "model_architecture": "SwinV2 Hybrid",
            "batch_size": "unknown",
            "learning_rate": "unknown",
            "epochs": "unknown",
        }

        if self.config:
            # Extract dataset information - check multiple possible paths
            data_config = None
            training_config = None

            # Try different possible paths for data configuration
            possible_data_paths = [
                ["experiments", "swinv2_hybrid", "data"],
                ["experiments", "swinv2_hybrid_cfd_320x320", "data"],
                ["data"],
            ]

            for path_keys in possible_data_paths:
                try:
                    current = self.config
                    for key in path_keys:
                        if hasattr(current, key):
                            current = getattr(current, key)
                        else:
                            raise AttributeError(f"Key {key} not found")
                    data_config = current
                    break
                except (AttributeError, KeyError):
                    continue

            # Try different possible paths for training configuration
            possible_training_paths = [
                ["experiments", "swinv2_hybrid", "training"],
                ["experiments", "swinv2_hybrid_cfd_320x320", "training"],
                ["training"],
            ]

            for path_keys in possible_training_paths:
                try:
                    current = self.config
                    for key in path_keys:
                        if hasattr(current, key):
                            current = getattr(current, key)
                        else:
                            raise AttributeError(f"Key {key} not found")
                    training_config = current
                    break
                except (AttributeError, KeyError):
                    continue

            # Extract dataset information
            if data_config and hasattr(data_config, "root_dir"):
                data_path = str(data_config.root_dir)
                if "CFD" in data_path:
                    metadata["dataset"] = "CFD"
                elif "crack500" in data_path:
                    metadata["dataset"] = "Crack500"
                elif "unified" in data_path:
                    metadata["dataset"] = "Unified"
                else:
                    metadata["dataset"] = (
                        data_path.split("/")[-1]
                        if "/" in data_path
                        else data_path
                    )

            # Extract image size
            if data_config and hasattr(data_config, "image_size"):
                image_size = data_config.image_size
                if isinstance(image_size, list) and len(image_size) >= 2:
                    metadata["image_size"] = f"{image_size[0]}x{image_size[1]}"
                else:
                    metadata["image_size"] = str(image_size)

            # Extract training parameters
            if training_config:
                if hasattr(training_config, "batch_size"):
                    metadata["batch_size"] = training_config.batch_size
                if hasattr(training_config, "learning_rate"):
                    metadata["learning_rate"] = training_config.learning_rate
                if hasattr(training_config, "epochs"):
                    metadata["epochs"] = training_config.epochs

        # If we have results, try to extract metadata from there
        if self.results and isinstance(self.results, dict):
            if "experiment_info" in self.results:
                exp_info = self.results["experiment_info"]
                if "total_epochs" in exp_info:
                    metadata["epochs"] = exp_info["total_epochs"]

                # Try to extract dataset info from experiment directory name
                exp_dir = exp_info.get("directory", "")
                if "cfd" in exp_dir.lower():
                    metadata["dataset"] = "CFD"
                elif "crack500" in exp_dir.lower():
                    metadata["dataset"] = "Crack500"
                elif "unified" in exp_dir.lower():
                    metadata["dataset"] = "Unified"

        return metadata

    def analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze memory usage patterns and provide recommendations."""
        logging.info("Analyzing memory usage...")

        memory_analysis = {
            "current_usage": {},
            "recommendations": [],
            "optimizations": [],
            "dataset_specific": {},
        }

        if torch.cuda.is_available():
            # Get current memory stats
            memory_stats = get_gpu_memory_usage()
            memory_analysis["current_usage"] = memory_stats

            # Analyze memory efficiency
            total_memory = memory_stats["total"] / 1024  # Convert MB to GB
            allocated_memory = (
                memory_stats["allocated"] / 1024
            )  # Convert MB to GB
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
            if memory_stats.get("allocated", 0) / 1024 > total_memory * 0.95:
                memory_analysis["recommendations"].append(
                    "Memory usage peaked near capacity. Monitor for potential memory leaks."
                )

            # Dataset-specific optimizations
            if self.experiment_metadata["dataset"] == "CFD":
                memory_analysis["dataset_specific"]["cfd"] = [
                    "CFD dataset (320x320): Consider increasing batch size for better GPU utilization",
                    "Smaller images allow for larger effective batch sizes",
                ]
            elif self.experiment_metadata["dataset"] == "Crack500":
                memory_analysis["dataset_specific"]["crack500"] = [
                    "Crack500 dataset: Monitor for class imbalance effects on memory",
                    "Consider focal loss for better handling of sparse crack pixels",
                ]

            # Optimization suggestions based on image size
            if self.config:
                # Try to get batch_size from different locations
                batch_size = 8  # default
                if hasattr(self.config, "training") and hasattr(
                    self.config.training, "batch_size"
                ):
                    batch_size = self.config.training.batch_size
                elif hasattr(self.config, "experiments"):
                    if hasattr(self.config.experiments, "swinv2_hybrid"):
                        if hasattr(
                            self.config.experiments.swinv2_hybrid, "training"
                        ):
                            batch_size = (
                                self.config.experiments.swinv2_hybrid.training.batch_size
                            )

                # Try to get image_size from different locations
                image_size = 256  # default
                if hasattr(self.config, "data") and hasattr(
                    self.config.data, "image_size"
                ):
                    img_size = self.config.data.image_size
                    # Handle ListConfig or list types
                    if hasattr(
                        img_size, "__getitem__"
                    ):  # Check if it's indexable
                        try:
                            image_size = int(img_size[0])
                        except (TypeError, ValueError):
                            image_size = 256
                    else:
                        image_size = int(img_size) if img_size else 256
                elif hasattr(self.config, "experiments"):
                    if hasattr(self.config.experiments, "swinv2_hybrid"):
                        if hasattr(
                            self.config.experiments.swinv2_hybrid, "data"
                        ):
                            img_size = (
                                self.config.experiments.swinv2_hybrid.data.image_size
                            )
                            # Handle ListConfig or list types
                            if hasattr(
                                img_size, "__getitem__"
                            ):  # Check if it's indexable
                                try:
                                    image_size = int(img_size[0])
                                except (TypeError, ValueError):
                                    image_size = 256
                            else:
                                image_size = int(img_size) if img_size else 256

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
        # This is a simplified calculation for SwinV2 Hybrid
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
            "dataset_specific_analysis": {},
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

                # Dataset-specific analysis
                progress_analysis["dataset_specific_analysis"] = (
                    self._analyze_dataset_specific_progress(metrics_data)
                )

        return progress_analysis

    def _parse_training_log(self) -> dict[str, Any]:
        """Parse training log file for metrics."""
        import re

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
            with open(self.log_file, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            epoch_pattern = re.compile(r"Epoch\s+(\d+)")
            loss_pattern = re.compile(r"loss:\s*([\d.]+)")
            val_loss_pattern = re.compile(r"val_loss:\s*([\d.]+)")
            val_iou_pattern = re.compile(r"val_iou:\s*([\d.]+)")
            val_dice_pattern = re.compile(r"val_dice:\s*([\d.]+)")
            time_pattern = re.compile(r"Total time:\s*([\d.]+)\s*hours?")

            current_epoch = 0
            for line in lines:
                # Parse epoch information
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    metrics_data["epochs"].append(current_epoch)

                # Parse training loss
                if "train" in line.lower():
                    loss_match = loss_pattern.search(line)
                    if loss_match:
                        metrics_data["train_loss"].append(
                            float(loss_match.group(1))
                        )

                # Parse validation metrics
                if "val" in line.lower() or "validation" in line.lower():
                    val_loss_match = val_loss_pattern.search(line)
                    val_iou_match = val_iou_pattern.search(line)
                    val_dice_match = val_dice_pattern.search(line)

                    if val_loss_match:
                        metrics_data["val_loss"].append(
                            float(val_loss_match.group(1))
                        )
                    if val_iou_match:
                        metrics_data["val_iou"].append(
                            float(val_iou_match.group(1))
                        )
                    if val_dice_match:
                        metrics_data["val_dice"].append(
                            float(val_dice_match.group(1))
                        )

                # Parse timing information
                time_match = time_pattern.search(line)
                if time_match:
                    metrics_data["total_time"] = float(time_match.group(1))
                elif "Training completed" in line:
                    # Try to extract time from this line
                    time_in_line = re.search(r"(\d+\.?\d*)\s*hours?", line)
                    if time_in_line:
                        metrics_data["total_time"] = float(
                            time_in_line.group(1)
                        )

            # Extract best metrics if available
            if metrics_data["val_iou"]:
                best_iou_idx = np.argmax(metrics_data["val_iou"])
                metrics_data["best_metrics"]["iou"] = metrics_data["val_iou"][
                    best_iou_idx
                ]
                metrics_data["best_metrics"]["epoch"] = (
                    metrics_data["epochs"][best_iou_idx]
                    if metrics_data["epochs"]
                    else best_iou_idx + 1
                )

            if metrics_data["val_dice"]:
                best_dice_idx = np.argmax(metrics_data["val_dice"])
                metrics_data["best_metrics"]["dice"] = metrics_data[
                    "val_dice"
                ][best_dice_idx]

            if metrics_data["val_loss"]:
                best_loss_idx = np.argmin(metrics_data["val_loss"])
                metrics_data["best_metrics"]["loss"] = metrics_data[
                    "val_loss"
                ][best_loss_idx]

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

    def _analyze_dataset_specific_progress(
        self, metrics_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze training progress specific to the dataset."""
        dataset_analysis = {}

        if self.experiment_metadata["dataset"] == "CFD":
            dataset_analysis["cfd"] = {
                "characteristics": [
                    "Smaller images (320x320) may converge faster",
                    "Less complex crack patterns compared to larger datasets",
                    "May benefit from higher learning rates",
                ],
                "recommendations": [
                    "Monitor for overfitting due to smaller dataset size",
                    "Consider data augmentation for better generalization",
                ],
            }
        elif self.experiment_metadata["dataset"] == "Crack500":
            dataset_analysis["crack500"] = {
                "characteristics": [
                    "Larger images with more complex crack patterns",
                    "Higher computational requirements",
                    "May require longer training for convergence",
                ],
                "recommendations": [
                    "Ensure sufficient training epochs for complex patterns",
                    "Monitor memory usage with larger images",
                ],
            }

        return dataset_analysis

    def compare_with_baselines(self) -> dict[str, Any]:
        """Compare experiment results with baseline models."""
        logging.info("Comparing with baseline models...")

        if not self.results:
            return {"error": "No results available for comparison"}

        # Define baseline results for crack segmentation
        # These are general baselines - dataset-specific comparisons would be more accurate
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
            "dataset_context": self.experiment_metadata["dataset"],
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
        report.append("SwinV2 Hybrid Experiment Analysis Report")
        report.append("=" * 80)
        report.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append(f"Experiment Directory: {self.experiment_dir}")
        report.append("")

        # Experiment metadata
        report.append("EXPERIMENT METADATA")
        report.append("-" * 40)
        for key, value in self.experiment_metadata.items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
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

        # Dataset-specific memory analysis
        if memory_analysis["dataset_specific"]:
            report.append("Dataset-Specific Analysis:")
            for dataset, recommendations in memory_analysis[
                "dataset_specific"
            ].items():
                report.append(f"  {dataset.upper()}:")
                for rec in recommendations:
                    report.append(f"    - {rec}")
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

        # Dataset-specific progress analysis
        if progress_analysis["dataset_specific_analysis"]:
            report.append("Dataset-Specific Progress:")
            for dataset, analysis in progress_analysis[
                "dataset_specific_analysis"
            ].items():
                report.append(f"  {dataset.upper()}:")
                if "characteristics" in analysis:
                    report.append("    Characteristics:")
                    for char in analysis["characteristics"]:
                        report.append(f"      - {char}")
                if "recommendations" in analysis:
                    report.append("    Recommendations:")
                    for rec in analysis["recommendations"]:
                        report.append(f"      - {rec}")
        report.append("")

        # Baseline comparison
        comparison = self.compare_with_baselines()
        report.append("BASELINE COMPARISON")
        report.append("-" * 40)
        report.append(
            f"Dataset Context: {comparison.get('dataset_context', 'Unknown')}"
        )
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

        # Check batch size recommendation
        batch_size = self.experiment_metadata.get("batch_size", "unknown")
        if (
            batch_size != "unknown"
            and isinstance(batch_size, int | float)
            and batch_size < 4
        ):
            report.append("- Consider increasing batch size if memory allows")

        # Dataset-specific recommendations
        if self.experiment_metadata["dataset"] == "CFD":
            report.append(
                "- CFD dataset: Consider data augmentation for better generalization"
            )
            report.append(
                "- Monitor for overfitting due to smaller dataset size"
            )
        elif self.experiment_metadata["dataset"] == "Crack500":
            report.append(
                "- Crack500 dataset: Ensure sufficient training epochs for complex patterns"
            )
            report.append("- Monitor memory usage with larger images")

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
        description="Analyze SwinV2 hybrid experiment (generic version)"
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
    analyzer = GenericExperimentAnalyzer(args.experiment_dir)

    # Generate and save analysis
    output_file = Path(args.output) if args.output else None
    analyzer.save_analysis(output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Experiment metadata summary
    print("Experiment Metadata:")
    for key, value in analyzer.experiment_metadata.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

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
            if isinstance(value, int | float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
