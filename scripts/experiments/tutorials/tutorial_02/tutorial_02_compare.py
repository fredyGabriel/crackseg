#!/usr/bin/env python3
"""
Tutorial 02: Experiment Comparison

This script demonstrates how to compare multiple experiments and analyze
their results using the comparison utilities.
"""

import argparse

# Add project root to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_experiment_data(experiment_paths: list[Path]) -> list[dict]:
    """Load experiment data from paths."""
    experiments_data = []

    for exp_path in experiment_paths:
        try:
            # Load metrics from experiment directory
            metrics_file = exp_path / "metrics" / "final_metrics.json"
            if metrics_file.exists():
                import json

                with open(metrics_file) as f:
                    metrics = json.load(f)

                experiments_data.append(
                    {
                        "name": exp_path.name,
                        "path": str(exp_path),
                        "metrics": metrics,
                        "timestamp": (
                            exp_path.name.split("-")[0]
                            if "-" in exp_path.name
                            else "unknown"
                        ),
                    }
                )
            else:
                print(f"Warning: No metrics file found in {exp_path}")

        except Exception as e:
            print(f"Error loading experiment {exp_path}: {e}")

    return experiments_data


def create_comparison_visualizations(
    experiments_data: list[dict], output_dir: Path
):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics for comparison
    metrics_data = {}
    for exp in experiments_data:
        metrics_data[exp["name"]] = exp["metrics"]

    # 1. Metrics comparison chart
    print("Creating metrics comparison chart...")
    _fig, ax = plt.subplots(figsize=(12, 8))

    metrics_names = ["iou", "dice", "precision", "recall", "f1"]
    x = np.arange(len(metrics_names))
    width = 0.8 / len(experiments_data)

    for i, (exp_name, metrics) in enumerate(metrics_data.items()):
        values = [metrics.get(metric, 0) for metric in metrics_names]
        ax.bar(x + i * width, values, width, label=exp_name, alpha=0.8)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Experiment Metrics Comparison")
    ax.set_xticks(x + width * (len(experiments_data) - 1) / 2)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 2. Training curves comparison
    print("Creating training curves comparison...")
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for exp_name, metrics in metrics_data.items():
        if "training_history" in metrics:
            history = metrics["training_history"]
            epochs = range(1, len(history.get("train_loss", [])) + 1)

            ax1.plot(
                epochs,
                history.get("train_loss", []),
                label=f"{exp_name} (Train)",
                alpha=0.8,
            )
            ax1.plot(
                epochs,
                history.get("val_loss", []),
                label=f"{exp_name} (Val)",
                linestyle="--",
                alpha=0.8,
            )

            ax2.plot(
                epochs,
                history.get("val_iou", []),
                label=f"{exp_name} (IoU)",
                alpha=0.8,
            )
            ax2.plot(
                epochs,
                history.get("val_dice", []),
                label=f"{exp_name} (Dice)",
                linestyle="--",
                alpha=0.8,
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "training_curves_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Create comparison table
    print("Creating comparison table...")
    comparison_df = pd.DataFrame(metrics_data).T

    # Select key metrics
    key_metrics = ["iou", "dice", "precision", "recall", "f1", "accuracy"]
    comparison_df = comparison_df[key_metrics]

    # Save as CSV
    comparison_df.to_csv(output_dir / "comparison_table.csv")

    # Create formatted table
    _fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    table_data.append(["Experiment"] + key_metrics)

    for exp_name, row in comparison_df.iterrows():
        table_data.append(
            [exp_name] + [f"{row[metric]:.4f}" for metric in key_metrics]
        )

    table = ax.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Experiment Comparison Table")
    plt.savefig(
        output_dir / "comparison_table.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"✅ Comparison visualizations saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Experiment comparison tutorial"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="List of experiment paths to compare",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/global/reports/tutorial_02_comparison",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # Convert to Path objects
    experiment_paths = [Path(exp) for exp in args.experiments]
    output_dir = Path(args.output_dir)

    print("🔍 Tutorial 02: Experiment Comparison")
    print("=" * 50)

    # Verify experiment paths exist
    for exp_path in experiment_paths:
        if not exp_path.exists():
            print(f"❌ Experiment path not found: {exp_path}")
            return

    print(f"📋 Comparing {len(experiment_paths)} experiments:")
    for exp_path in experiment_paths:
        print(f"   • {exp_path.name}")

    # Load experiment data
    experiments_data = load_experiment_data(experiment_paths)

    if not experiments_data:
        print("❌ No experiment data could be loaded!")
        return

    print(f"✅ Loaded data for {len(experiments_data)} experiments")

    # Create comparison visualizations
    create_comparison_visualizations(experiments_data, output_dir)

    # Print summary
    print("\n📊 Comparison Summary:")
    for exp in experiments_data:
        metrics = exp["metrics"]
        print(f"   {exp['name']}:")
        print(f"     IoU: {metrics.get('iou', 'N/A'):.4f}")
        print(f"     Dice: {metrics.get('dice', 'N/A'):.4f}")
        print(f"     Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"     Recall: {metrics.get('recall', 'N/A'):.4f}")

    print(f"\n📁 Results saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
