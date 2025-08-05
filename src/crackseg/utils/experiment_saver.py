"""Experiment data saver for evaluation/reporting compatibility.

This module provides a standardized way to save experiment data that is
compatible with the evaluation/ and reporting/ modules.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import DictConfig


class ExperimentDataSaver:
    """Save experiment data in a format compatible with evaluation/reporting modules."""

    def __init__(self, experiment_dir: Path) -> None:
        """Initialize the experiment data saver.

        Args:
            experiment_dir: Path to the experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logging.getLogger(__name__)

    def save_complete_summary(
        self,
        experiment_config: DictConfig,
        final_metrics: dict[str, float],
        best_epoch: int,
        training_time: float,
    ) -> Path:
        """Save complete experiment summary.

        Args:
            experiment_config: Experiment configuration
            final_metrics: Final validation metrics
            best_epoch: Best epoch number
            training_time: Total training time in seconds

        Returns:
            Path to the saved summary file
        """
        # Create metrics directory
        metrics_dir = self.experiment_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Create complete summary with safe attribute access
        complete_summary = {
            "experiment_name": experiment_config.experiment.name,
            "project_name": experiment_config.project_name,
            "model_type": experiment_config.model._target_,
            "dataset": getattr(
                experiment_config.data, "dataset_path", "unknown"
            ),
            "training_epochs": experiment_config.training.epochs,
            "batch_size": experiment_config.training.batch_size,
            "learning_rate": experiment_config.training.learning_rate,
            "final_metrics": final_metrics,
            "best_epoch": best_epoch,
            "total_training_time": training_time,
            "timestamp": datetime.now().isoformat(),
            "hardware_info": {
                "device": str(self._get_device()),
                "cuda_available": self._is_cuda_available(),
                "gpu_name": self._get_gpu_name(),
            },
        }

        # Save summary
        summary_path = metrics_dir / "complete_summary.json"
        with open(summary_path, "w") as f:
            json.dump(complete_summary, f, indent=2, default=str)

        self.logger.info(f"Complete summary saved to: {summary_path}")
        return summary_path

    def save_per_epoch_metrics(
        self,
        experiment_config: DictConfig,
        train_losses: dict[int, float],
        val_losses: dict[int, float],
        val_ious: dict[int, float],
        val_f1s: dict[int, float],
        val_precisions: dict[int, float],
        val_recalls: dict[int, float],
        val_dices: dict[int, float],
        val_accuracies: dict[int, float],
    ) -> Path:
        """Save per-epoch metrics in JSONL format.

        Args:
            experiment_config: Experiment configuration
            train_losses: Training losses by epoch
            val_losses: Validation losses by epoch
            val_ious: Validation IoU by epoch
            val_f1s: Validation F1 by epoch
            val_precisions: Validation precision by epoch
            val_recalls: Validation recall by epoch
            val_dices: Validation dice by epoch
            val_accuracies: Validation accuracy by epoch

        Returns:
            Path to the saved metrics file
        """
        # Create metrics directory
        metrics_dir = self.experiment_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Save per-epoch metrics in JSONL format
        metrics_jsonl_path = metrics_dir / "metrics.jsonl"
        with open(metrics_jsonl_path, "w") as f:
            for epoch in range(1, experiment_config.training.epochs + 1):
                epoch_data = {
                    "epoch": epoch,
                    "train_loss": train_losses.get(epoch, 0.0),
                    "val_loss": val_losses.get(epoch, 0.0),
                    "val_iou": val_ious.get(epoch, 0.0),
                    "val_f1": val_f1s.get(epoch, 0.0),
                    "val_precision": val_precisions.get(epoch, 0.0),
                    "val_recall": val_recalls.get(epoch, 0.0),
                    "val_dice": val_dices.get(epoch, 0.0),
                    "val_accuracy": val_accuracies.get(epoch, 0.0),
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(epoch_data) + "\n")

        self.logger.info(f"Per-epoch metrics saved to: {metrics_jsonl_path}")
        return metrics_jsonl_path

    def save_validation_metrics(
        self,
        final_metrics: dict[str, float],
        best_epoch: int,
        best_metrics: dict[str, float],
    ) -> Path:
        """Save validation metrics separately.

        Args:
            final_metrics: Final validation metrics
            best_epoch: Best epoch number
            best_metrics: Best epoch metrics

        Returns:
            Path to the saved validation metrics file
        """
        # Create metrics directory
        metrics_dir = self.experiment_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Save validation metrics
        validation_metrics = {
            "final_validation": final_metrics,
            "best_validation": {
                "epoch": best_epoch,
                **best_metrics,
            },
        }

        validation_metrics_path = metrics_dir / "validation_metrics.json"
        with open(validation_metrics_path, "w") as f:
            json.dump(validation_metrics, f, indent=2)

        self.logger.info(
            f"Validation metrics saved to: {validation_metrics_path}"
        )
        return validation_metrics_path

    def save_training_logs(self, log_file: str = "training.log") -> Path:
        """Save training logs to the experiment directory.

        Args:
            log_file: Name of the log file to copy

        Returns:
            Path to the saved log file
        """
        # Create logs directory
        logs_dir = self.experiment_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Copy the current log to logs directory
        current_log = Path(log_file)
        if current_log.exists():
            log_path = logs_dir / log_file
            shutil.copy2(current_log, log_path)
            self.logger.info(f"Training logs saved to: {log_path}")
            return log_path
        else:
            self.logger.warning(f"Log file not found: {log_file}")
            return logs_dir / log_file

    def save_final_results_yaml(self, final_metrics: dict[str, float]) -> Path:
        """Save final results in YAML format for backward compatibility.

        Args:
            final_metrics: Final validation metrics

        Returns:
            Path to the saved final results file
        """
        final_results_path = self.experiment_dir / "final_results.yaml"
        with open(final_results_path, "w") as f:
            yaml.dump(final_metrics, f, default_flow_style=False)

        self.logger.info(f"Final results saved to: {final_results_path}")
        return final_results_path

    def save_all_experiment_data(
        self,
        experiment_config: DictConfig,
        final_metrics: dict[str, float],
        best_epoch: int,
        training_time: float,
        train_losses: dict[int, float],
        val_losses: dict[int, float],
        val_ious: dict[int, float],
        val_f1s: dict[int, float],
        val_precisions: dict[int, float],
        val_recalls: dict[int, float],
        val_dices: dict[int, float],
        val_accuracies: dict[int, float],
        best_metrics: dict[str, float],
        log_file: str = "training.log",
    ) -> dict[str, Path]:
        """Save all experiment data with evaluation/reporting compatibility.

        Args:
            experiment_config: Experiment configuration
            final_metrics: Final validation metrics
            best_epoch: Best epoch number
            training_time: Total training time in seconds
            train_losses: Training losses by epoch
            val_losses: Validation losses by epoch
            val_ious: Validation IoU by epoch
            val_f1s: Validation F1 by epoch
            val_precisions: Validation precision by epoch
            val_recalls: Validation recall by epoch
            val_dices: Validation dice by epoch
            val_accuracies: Validation accuracy by epoch
            best_metrics: Best epoch metrics
            log_file: Name of the log file to copy

        Returns:
            Dictionary mapping file types to their paths
        """
        self.logger.info(
            "Saving experiment data with evaluation/reporting compatibility..."
        )

        # Save all data
        summary_path = self.save_complete_summary(
            experiment_config, final_metrics, best_epoch, training_time
        )
        metrics_path = self.save_per_epoch_metrics(
            experiment_config,
            train_losses,
            val_losses,
            val_ious,
            val_f1s,
            val_precisions,
            val_recalls,
            val_dices,
            val_accuracies,
        )
        validation_path = self.save_validation_metrics(
            final_metrics, best_epoch, best_metrics
        )
        logs_path = self.save_training_logs(log_file)
        final_results_path = self.save_final_results_yaml(final_metrics)

        # Log success message (without emojis for Windows compatibility)
        self.logger.info(
            "Experiment data saved with evaluation/reporting compatibility:"
        )
        self.logger.info(f"  Complete summary: {summary_path}")
        self.logger.info(f"  Per-epoch metrics: {metrics_path}")
        self.logger.info(f"  Validation metrics: {validation_path}")
        self.logger.info(f"  Training logs: {logs_path}")
        self.logger.info(f"  Final results: {final_results_path}")

        return {
            "summary": summary_path,
            "metrics": metrics_path,
            "validation": validation_path,
            "logs": logs_path,
            "final_results": final_results_path,
        }

    def _get_device(self) -> str:
        """Get current device string."""
        try:
            import torch

            return str(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        except ImportError:
            return "unknown"

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_name(self) -> str:
        """Get GPU name."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            else:
                return "CPU"
        except ImportError:
            return "unknown"


def save_experiment_data(
    experiment_dir: Path,
    experiment_config: DictConfig,
    final_metrics: dict[str, float],
    best_epoch: int,
    training_time: float,
    train_losses: dict[int, float],
    val_losses: dict[int, float],
    val_ious: dict[int, float],
    val_f1s: dict[int, float],
    val_precisions: dict[int, float],
    val_recalls: dict[int, float],
    val_dices: dict[int, float],
    val_accuracies: dict[int, float],
    best_metrics: dict[str, float],
    log_file: str = "training.log",
) -> dict[str, Path]:
    """Convenience function to save all experiment data.

    Args:
        experiment_dir: Path to experiment directory
        experiment_config: Experiment configuration
        final_metrics: Final validation metrics
        best_epoch: Best epoch number
        training_time: Total training time in seconds
        train_losses: Training losses by epoch
        val_losses: Validation losses by epoch
        val_ious: Validation IoU by epoch
        val_f1s: Validation F1 by epoch
        val_precisions: Validation precision by epoch
        val_recalls: Validation recall by epoch
        val_dices: Validation dice by epoch
        val_accuracies: Validation accuracy by epoch
        best_metrics: Best epoch metrics
        log_file: Name of the log file to copy

    Returns:
        Dictionary mapping file types to their paths
    """
    saver = ExperimentDataSaver(experiment_dir)
    return saver.save_all_experiment_data(
        experiment_config=experiment_config,
        final_metrics=final_metrics,
        best_epoch=best_epoch,
        training_time=training_time,
        train_losses=train_losses,
        val_losses=val_losses,
        val_ious=val_ious,
        val_f1s=val_f1s,
        val_precisions=val_precisions,
        val_recalls=val_recalls,
        val_dices=val_dices,
        val_accuracies=val_accuracies,
        best_metrics=best_metrics,
        log_file=log_file,
    )
