#!/usr/bin/env python3
"""
Tutorial 02 Experiment Comparison Script

This script compares the results of experiments created in Tutorial 02:
"Creating Custom Experiments (CLI Only)"

Reference: docs/tutorials/02_custom_experiment_cli.md

Usage:
    python scripts/experiments/tutorial_02_compare.py
"""

import json
from pathlib import Path


def load_metrics(experiment_dir):
    """Load final metrics for an experiment."""
    metrics_file = experiment_dir / "metrics" / "complete_summary.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def main():
    # Define experiment directories for Tutorial 02 experiments
    experiments = {
        "high_lr_experiment": (
            "src/crackseg/outputs/experiments/20250723-005521-default"
        ),
        "low_lr_experiment": (
            "src/crackseg/outputs/experiments/20250723-005704-default"
        ),
        "medium_lr_experiment": (
            "src/crackseg/outputs/experiments/20250723-010032-default"
        ),
    }

    print("Tutorial 02 - Experiment Comparison Results")
    print("=" * 60)
    print("Reference: docs/tutorials/02_custom_experiment_cli.md")
    print("=" * 60)

    for exp_name, exp_dir in experiments.items():
        metrics = load_metrics(Path(exp_dir))
        if metrics:
            best_metrics = metrics.get("best_metrics", {})
            print(f"\n{exp_name}:")
            loss_value = best_metrics.get("loss", {}).get("value", "N/A")
            print(f"  Final Loss: {loss_value:.4f}")
            iou_value = best_metrics.get("iou", {}).get("value", "N/A")
            print(f"  Final IoU: {iou_value:.4f}")
            f1_value = best_metrics.get("f1", {}).get("value", "N/A")
            print(f"  Final F1: {f1_value:.4f}")
            precision_value = best_metrics.get("precision", {}).get(
                "value", "N/A"
            )
            print(f"  Final Precision: {precision_value:.4f}")
            recall_value = best_metrics.get("recall", {}).get("value", "N/A")
            print(f"  Final Recall: {recall_value:.4f}")
            total_epochs = metrics.get("experiment_info", {}).get(
                "total_epochs", "N/A"
            )
            print(f"  Total Epochs: {total_epochs}")
        else:
            print(f"\n{exp_name}: No metrics found")

    print("\n" + "=" * 60)
    print("Tutorial 02 Summary:")
    print("- High LR experiment: Learning rate 0.001, 50 epochs")
    print("- Low LR experiment: Learning rate 0.00001, 100 epochs")
    print("- Medium LR experiment: Learning rate 0.0001, 75 epochs")
    print("\nConfiguration files: configs/experiments/tutorial_02/")
    print("Scripts: scripts/experiments/tutorial_02_*.py")


if __name__ == "__main__":
    main()
