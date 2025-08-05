#!/usr/bin/env python3
"""
Example of how to use the ExperimentDataSaver in any experiment script.

This demonstrates how to integrate the standardized experiment data saving
into any experiment script for compatibility with evaluation/ and reporting/ modules.
"""

import logging

# Add project root to path
import sys
import time
from pathlib import Path

from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.utils.experiment_saver import save_experiment_data  # noqa: E402


def example_experiment_with_saver():
    """Example of how to use the experiment saver in any experiment."""

    # Simulate experiment configuration
    experiment_config = DictConfig(
        {
            "experiment": {"name": "example_experiment"},
            "project_name": "crack-segmentation",
            "model": {"_target_": "crackseg.model.SwinV2CnnAsppUNet"},
            "data": {"dataset_path": "data/unified"},
            "training": {
                "epochs": 10,
                "batch_size": 4,
                "learning_rate": 1e-4,
            },
        }
    )

    # Simulate experiment results
    final_metrics = {
        "val_precision": 0.85,
        "val_recall": 0.82,
        "val_f1": 0.83,
        "val_iou": 0.71,
        "val_dice": 0.83,
        "val_accuracy": 0.95,
        "val_loss": 0.25,
    }

    # Simulate per-epoch metrics
    train_losses = {i: 0.5 - i * 0.02 for i in range(1, 11)}
    val_losses = {i: 0.4 - i * 0.015 for i in range(1, 11)}
    val_ious = {i: 0.6 + i * 0.01 for i in range(1, 11)}
    val_f1s = {i: 0.75 + i * 0.008 for i in range(1, 11)}
    val_precisions = {i: 0.8 + i * 0.005 for i in range(1, 11)}
    val_recalls = {i: 0.75 + i * 0.007 for i in range(1, 11)}
    val_dices = {i: 0.77 + i * 0.006 for i in range(1, 11)}
    val_accuracies = {i: 0.92 + i * 0.003 for i in range(1, 11)}

    # Simulate best epoch
    best_epoch = 8
    best_metrics = {
        "precision": 0.84,
        "recall": 0.81,
        "f1": 0.82,
        "iou": 0.70,
        "dice": 0.82,
        "accuracy": 0.94,
        "loss": 0.26,
    }

    # Create experiment directory
    experiment_dir = Path("artifacts/experiments/example_experiment")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save all experiment data with evaluation/reporting compatibility
    start_time = time.time()
    training_time = time.time() - start_time

    saved_files = save_experiment_data(
        experiment_dir=experiment_dir,
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
        log_file="example_experiment.log",
    )

    print("✅ Experiment data saved successfully!")
    print("📁 Files created:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")

    return saved_files


def example_integration_in_trainer():
    """Example of how to integrate the saver into a trainer class."""

    class ExampleTrainer:
        """Example trainer that uses the experiment saver."""

        def __init__(self, config: DictConfig):
            self.config = config
            self.start_time = time.time()
            self.best_epoch = 0
            self.best_metrics = {}

            # Track metrics during training
            self.train_losses = {}
            self.val_losses = {}
            self.val_ious = {}
            self.val_f1s = {}
            self.val_precisions = {}
            self.val_recalls = {}
            self.val_dices = {}
            self.val_accuracies = {}

        def train_epoch(self, epoch: int):
            """Simulate training one epoch."""
            # Simulate training
            train_loss = 0.5 - epoch * 0.02
            val_loss = 0.4 - epoch * 0.015
            val_iou = 0.6 + epoch * 0.01
            val_f1 = 0.75 + epoch * 0.008

            # Store metrics
            self.train_losses[epoch] = train_loss
            self.val_losses[epoch] = val_loss
            self.val_ious[epoch] = val_iou
            self.val_f1s[epoch] = val_f1
            self.val_precisions[epoch] = 0.8 + epoch * 0.005
            self.val_recalls[epoch] = 0.75 + epoch * 0.007
            self.val_dices[epoch] = 0.77 + epoch * 0.006
            self.val_accuracies[epoch] = 0.92 + epoch * 0.003

            # Update best metrics
            if val_iou > self.best_metrics.get("iou", 0):
                self.best_epoch = epoch
                self.best_metrics = {
                    "precision": self.val_precisions[epoch],
                    "recall": self.val_recalls[epoch],
                    "f1": self.val_f1s[epoch],
                    "iou": self.val_ious[epoch],
                    "dice": self.val_dices[epoch],
                    "accuracy": self.val_accuracies[epoch],
                    "loss": self.val_losses[epoch],
                }

            return {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_f1": val_f1,
            }

        def save_experiment_data(
            self, experiment_dir: Path, final_metrics: dict[str, float]
        ):
            """Save experiment data using the standardized saver."""
            training_time = time.time() - self.start_time

            return save_experiment_data(
                experiment_dir=experiment_dir,
                experiment_config=self.config,
                final_metrics=final_metrics,
                best_epoch=self.best_epoch,
                training_time=training_time,
                train_losses=self.train_losses,
                val_losses=self.val_losses,
                val_ious=self.val_ious,
                val_f1s=self.val_f1s,
                val_precisions=self.val_precisions,
                val_recalls=self.val_recalls,
                val_dices=self.val_dices,
                val_accuracies=self.val_accuracies,
                best_metrics=self.best_metrics,
                log_file="example_trainer.log",
            )

    # Example usage
    config = DictConfig(
        {
            "experiment": {"name": "example_trainer"},
            "project_name": "crack-segmentation",
            "model": {"_target_": "crackseg.model.SwinV2CnnAsppUNet"},
            "data": {"dataset_path": "data/unified"},
            "training": {
                "epochs": 5,
                "batch_size": 4,
                "learning_rate": 1e-4,
            },
        }
    )

    trainer = ExampleTrainer(config)

    # Simulate training
    for epoch in range(1, 6):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: {metrics}")

    # Save experiment data
    experiment_dir = Path("artifacts/experiments/example_trainer")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = {
        "val_precision": 0.82,
        "val_recall": 0.79,
        "val_f1": 0.80,
        "val_iou": 0.67,
        "val_dice": 0.80,
        "val_accuracy": 0.93,
        "val_loss": 0.28,
    }

    saved_files = trainer.save_experiment_data(experiment_dir, final_metrics)

    print("✅ Trainer experiment data saved successfully!")
    return saved_files


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("EXPERIMENT SAVER EXAMPLE")
    print("=" * 60)

    # Run examples
    print("\n1. Basic example:")
    example_experiment_with_saver()

    print("\n2. Trainer integration example:")
    example_integration_in_trainer()

    print("\n✅ All examples completed successfully!")
    print("\n📋 Usage in your experiment scripts:")
    print(
        "1. Import: from crackseg.utils.experiment_saver import save_experiment_data"
    )
    print("2. Track metrics during training")
    print("3. Call save_experiment_data() at the end")
    print(
        "4. All data will be compatible with evaluation/ and reporting/ modules"
    )
