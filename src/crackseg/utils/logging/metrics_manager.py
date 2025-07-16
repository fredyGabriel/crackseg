#!/usr/bin/env python3
"""Unified metrics management for training and evaluation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from crackseg.utils.logging.base import get_logger


class MetricsManager:
    """Centralized manager for consistent metric logging and storage."""

    def __init__(
        self,
        experiment_dir: Path | str,
        logger: logging.Logger | None = None,
        config: DictConfig | None = None,
    ) -> None:
        """Initialize the MetricsManager.

        Args:
            experiment_dir: Base directory for the experiment
            logger: Optional Python logger instance for console output
            config: Optional configuration for metric settings
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger or get_logger(__name__)
        self.config = config or {}

        # Create standardized directory structure
        self.metrics_dir = self.experiment_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Standardized metric files
        self.training_metrics_file = (
            self.metrics_dir / "training_metrics.jsonl"
        )
        self.validation_metrics_file = (
            self.metrics_dir / "validation_metrics.jsonl"
        )
        self.test_metrics_file = self.metrics_dir / "test_metrics.jsonl"
        self.summary_file = self.metrics_dir / "summary.json"

        # Initialize summary tracking
        self._summary: dict[str, Any] = {
            "experiment_start": datetime.now().isoformat(),
            "total_epochs": 0,
            "best_metrics": {},
            "metric_history": [],
        }

    def log_training_metrics(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, float | int | torch.Tensor],
        phase: str = "train",
    ) -> None:
        """Log training metrics with standardized format.

        Args:
            epoch: Current epoch number
            step: Current step/batch number
            metrics: Dictionary of metric names and values
            phase: Training phase ('train', 'val', 'test')
        """
        # Convert tensors to floats
        scalar_metrics = self._extract_scalar_metrics(metrics)

        # Create standardized metric entry
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "phase": phase,
            "metrics": scalar_metrics,
        }

        # Determine appropriate file based on phase
        if phase == "train":
            target_file = self.training_metrics_file
        elif phase == "val":
            target_file = self.validation_metrics_file
        elif phase == "test":
            target_file = self.test_metrics_file
        else:
            target_file = self.training_metrics_file

        # Append to appropriate metrics file
        self._append_to_file(target_file, metric_entry)

        # Log to console
        formatted_metrics = self._format_metrics_for_console(scalar_metrics)
        self.logger.info(
            f"[{phase.upper()}] Epoch {epoch} Step {step} | "
            f"{formatted_metrics}"
        )

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float] | None = None,
        val_metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Log epoch summary with all available metrics.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
            learning_rate: Current learning rate
        """
        epoch_summary = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "train_metrics": train_metrics or {},
            "val_metrics": val_metrics or {},
            "learning_rate": learning_rate,
        }

        # Update running summary
        self._summary["total_epochs"] = max(
            self._summary["total_epochs"], epoch
        )
        self._summary["metric_history"].append(epoch_summary)

        # Update best metrics tracking
        if val_metrics:
            self._update_best_metrics(val_metrics, epoch)

        # Save updated summary
        self._save_summary()

        # Log epoch summary to console
        self._log_epoch_summary_to_console(
            epoch, train_metrics, val_metrics, learning_rate
        )

    def get_metric_history(
        self, metric_name: str, phase: str = "val"
    ) -> list[tuple[int, float]]:
        """Get historical values for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve
            phase: Phase to get metrics from ('train', 'val', 'test')

        Returns:
            List of (epoch, value) tuples
        """
        history: list[tuple[int, float]] = []

        # Determine file to read from
        if phase == "val":
            source_file = self.validation_metrics_file
        elif phase == "train":
            source_file = self.training_metrics_file
        elif phase == "test":
            source_file = self.test_metrics_file
        else:
            return history

        # Read and filter metrics
        if source_file.exists():
            try:
                with open(source_file, encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if metric_name in entry.get("metrics", {}):
                            epoch = entry.get("epoch", 0)
                            value = entry["metrics"][metric_name]
                            history.append((epoch, value))
            except (json.JSONDecodeError, KeyError, OSError) as e:
                self.logger.warning(f"Error reading metric history: {e}")

        return history

    def get_best_metric(self, metric_name: str) -> dict[str, Any] | None:
        """Get the best recorded value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with best value info or None if not found
        """
        return self._summary.get("best_metrics", {}).get(metric_name)

    def export_metrics_summary(
        self, output_path: Path | str | None = None
    ) -> Path:
        """Export a comprehensive metrics summary.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to the exported summary file
        """
        if output_path is None:
            output_path = self.metrics_dir / "complete_summary.json"
        else:
            output_path = Path(output_path)

        # Compile comprehensive summary
        complete_summary = {
            "experiment_info": {
                "directory": str(self.experiment_dir),
                "start_time": self._summary.get("experiment_start"),
                "total_epochs": self._summary.get("total_epochs", 0),
            },
            "best_metrics": self._summary.get("best_metrics", {}),
            "epoch_summaries": self._summary.get("metric_history", []),
            "available_metrics": self._get_available_metrics(),
        }

        # Save complete summary
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(complete_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics summary exported to: {output_path}")

        return output_path

    def _extract_scalar_metrics(
        self, metrics: dict[str, float | int | torch.Tensor]
    ) -> dict[str, float]:
        """Extract scalar float values from mixed metric types."""
        scalar_metrics: dict[str, float] = {}

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar_metrics[name] = value.item()
                else:
                    # For multi-element tensors, take mean
                    scalar_metrics[name] = value.mean().item()
            else:
                # value must be float | int
                scalar_metrics[name] = float(value)

        return scalar_metrics

    def _format_metrics_for_console(self, metrics: dict[str, float]) -> str:
        """Format metrics for readable console output."""
        formatted_parts: list[str] = []
        for name, value in metrics.items():
            formatted_parts.append(f"{name}: {value:.4f}")
        return " | ".join(formatted_parts)

    def _append_to_file(self, file_path: Path, entry: dict[str, Any]) -> None:
        """Append a JSON entry to a JSONL file."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            self.logger.error(f"Failed to write to {file_path}: {e}")

    def _update_best_metrics(
        self, val_metrics: dict[str, float], epoch: int
    ) -> None:
        """Update tracking of best metric values."""
        for metric_name, value in val_metrics.items():
            current_best = self._summary["best_metrics"].get(metric_name)

            # Determine if this is better (higher is better for most metrics
            # except loss)
            is_loss_metric = "loss" in metric_name.lower()
            is_better = (
                current_best is None
                or (is_loss_metric and value < current_best["value"])
                or (not is_loss_metric and value > current_best["value"])
            )

            if is_better:
                self._summary["best_metrics"][metric_name] = {
                    "value": value,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                }

    def _save_summary(self) -> None:
        """Save the current summary to disk."""
        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(self._summary, f, indent=2, ensure_ascii=False)
        except OSError as e:
            self.logger.error(f"Failed to save summary: {e}")

    def _log_epoch_summary_to_console(
        self,
        epoch: int,
        train_metrics: dict[str, float] | None,
        val_metrics: dict[str, float] | None,
        learning_rate: float | None,
    ) -> None:
        """Log epoch summary to console in a readable format."""
        self.logger.info(f"=== Epoch {epoch} Summary ===")

        if train_metrics:
            train_str = self._format_metrics_for_console(train_metrics)
            self.logger.info(f"Train: {train_str}")

        if val_metrics:
            val_str = self._format_metrics_for_console(val_metrics)
            self.logger.info(f"Val:   {val_str}")

        if learning_rate is not None:
            self.logger.info(f"LR:    {learning_rate:.6f}")

        # Show best metrics achieved so far
        best_info: list[str] = []
        for metric_name, best_data in self._summary["best_metrics"].items():
            best_info.append(
                f"{metric_name}: {best_data['value']:.4f} "
                f"(epoch {best_data['epoch']})"
            )

        if best_info:
            self.logger.info(f"Best:  {' | '.join(best_info)}")

    def _get_available_metrics(self) -> dict[str, list[str]]:
        """Get list of available metrics from each phase."""
        available: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for phase, file_path in [
            ("train", self.training_metrics_file),
            ("val", self.validation_metrics_file),
            ("test", self.test_metrics_file),
        ]:
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            metrics = entry.get("metrics", {})
                            for metric_name in metrics:
                                if metric_name not in available[phase]:
                                    available[phase].append(metric_name)
                except (json.JSONDecodeError, OSError):
                    continue

        return available
