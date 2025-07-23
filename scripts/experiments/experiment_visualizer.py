#!/usr/bin/env python3
"""
Generic Experiment Visualization Script

This script creates detailed visualizations and analysis of any set of
experiments. It generates plots, tables, and comprehensive analysis of the
results.

Usage:
    python scripts/experiments/experiment_visualizer.py --experiments exp1,
      exp2,exp3
    python scripts/experiments/experiment_visualizer.py --experiment-dirs
      path1,path2,path3
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_experiment_data(experiment_dir: Path) -> dict[str, Any]:
    """Load all data for an experiment including metrics and logs."""
    data = {}

    # Load complete summary
    summary_file = experiment_dir / "metrics" / "complete_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data["summary"] = json.load(f)

    # Load detailed metrics
    metrics_file = experiment_dir / "metrics" / "metrics.jsonl"
    if metrics_file.exists():
        metrics_data = []
        with open(metrics_file) as f:
            for line in f:
                metrics_data.append(json.loads(line.strip()))
        data["metrics"] = metrics_data

    return data


def create_comparison_table(experiments_data: dict[str, dict]) -> pd.DataFrame:
    """Create a comparison table of all experiments."""
    rows = []

    for exp_name, data in experiments_data.items():
        if "summary" in data:
            summary = data["summary"]
            best_metrics = summary.get("best_metrics", {})

            row = {
                "Experiment": exp_name,
                "Final Loss": best_metrics.get("loss", {}).get(
                    "value", np.nan
                ),
                "Final IoU": best_metrics.get("iou", {}).get("value", np.nan),
                "Final F1": best_metrics.get("f1", {}).get("value", np.nan),
                "Final Precision": best_metrics.get("precision", {}).get(
                    "value", np.nan
                ),
                "Final Recall": best_metrics.get("recall", {}).get(
                    "value", np.nan
                ),
                "Total Epochs": summary.get("experiment_info", {}).get(
                    "total_epochs", np.nan
                ),
                "Best Epoch": summary.get("experiment_info", {}).get(
                    "best_epoch", np.nan
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def plot_training_curves(
    experiments_data: dict[str, dict],
    title: str = "Training Curves Comparison",
    save_path: str | Path | None = None,
):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    metrics_to_plot = ["loss", "iou", "f1", "precision"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]

        for j, (exp_name, data) in enumerate(experiments_data.items()):
            if "metrics" in data and data["metrics"]:
                epochs = []
                values = []

                for entry in data["metrics"]:
                    if metric in entry:
                        epochs.append(entry.get("epoch", len(epochs) + 1))
                        values.append(entry[metric])

                if epochs and values:
                    ax.plot(
                        epochs,
                        values,
                        label=exp_name,
                        color=colors[j % len(colors)],
                        linewidth=2,
                        alpha=0.8,
                    )

        ax.set_title(f"{metric.upper()} Over Time", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Training curves saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_performance_radar(
    experiments_data: dict[str, dict],
    title: str = "Performance Comparison (Radar Chart)",
    save_path: str | Path | None = None,
):
    """Create a radar chart comparing final performance metrics."""
    # Prepare data for radar chart
    metrics = ["Final IoU", "Final F1", "Final Precision", "Final Recall"]
    comparison_df = create_comparison_table(experiments_data)

    if comparison_df.empty:
        print("❌ No data available for radar chart")
        return

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle

    # Initialize the plot
    _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Plot data for each experiment
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row["Experiment"],
            color=colors[i % len(colors)],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title(title, size=20, color="black", y=1.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Radar chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def print_detailed_analysis(
    comparison_df: pd.DataFrame, title: str = "EXPERIMENT ANALYSIS"
):
    """Print detailed analysis of experiment results."""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)

    if comparison_df.empty:
        print("❌ No data available for analysis")
        return

    # Print comparison table
    print("\n📋 COMPARISON TABLE:")
    print("-" * 60)
    print(comparison_df.to_string(index=False, float_format="%.4f"))

    # Print summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print("-" * 60)

    # Best performers
    if not comparison_df.empty:
        best_loss_exp = comparison_df.loc[comparison_df["Final Loss"].idxmin()]
        print(
            f"• Best Loss: {best_loss_exp['Experiment']} "
            f"({best_loss_exp['Final Loss']:.4f})"
        )

        best_iou_exp = comparison_df.loc[comparison_df["Final IoU"].idxmax()]
        print(
            f"• Best IoU: {best_iou_exp['Experiment']} "
            f"({best_iou_exp['Final IoU']:.4f})"
        )

        best_f1_exp = comparison_df.loc[comparison_df["Final F1"].idxmax()]
        print(
            f"• Best F1: {best_f1_exp['Experiment']} "
            f"({best_f1_exp['Final F1']:.4f})"
        )
        print(f"• Most stable: {best_f1_exp['Experiment']} (highest F1 score)")

    # Training efficiency
    print("\n⏱️ TRAINING EFFICIENCY:")
    print("-" * 60)
    if not comparison_df.empty:
        fastest_exp = comparison_df.loc[comparison_df["Total Epochs"].idxmin()]
        print(
            f"• Fastest convergence: {fastest_exp['Experiment']} "
            f"({fastest_exp['Total Epochs']:.0f} epochs)"
        )

        most_epochs_exp = comparison_df.loc[
            comparison_df["Total Epochs"].idxmax()
        ]
        print(
            f"• Longest training: {most_epochs_exp['Experiment']} "
            f"({most_epochs_exp['Total Epochs']:.0f} epochs)"
        )

    print("\n" + "=" * 60)


def find_experiment_directories(
    base_path: str = "src/crackseg/outputs/experiments",
) -> list[Path]:
    """Find all experiment directories in the base path."""
    base_dir = Path(base_path)
    if not base_dir.exists():
        return []

    # Find directories that contain metrics/complete_summary.json
    experiment_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir():
            metrics_file = item / "metrics" / "complete_summary.json"
            if metrics_file.exists():
                experiment_dirs.append(item)

    # Sort by modification time (newest first)
    experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return experiment_dirs


def main():
    """Main function to run the experiment visualizer."""
    parser = argparse.ArgumentParser(
        description="Visualize and analyze experiment results"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        help="Comma-separated list of experiment names to analyze",
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        help="Comma-separated list of experiment directory paths",
    )
    parser.add_argument(
        "--auto-find",
        action="store_true",
        help="Automatically find and analyze recent experiments",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Max experiments to analyze (when using --auto-find)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/reports/experiment_plots",
        help="Directory to save generated plots",
    )

    args = parser.parse_args()

    # Prepare experiments data
    experiments = {}

    if args.experiment_dirs:
        dir_paths = [Path(p.strip()) for p in args.experiment_dirs.split(",")]
        exp_names = [p.name for p in dir_paths]
        experiments = dict(zip(exp_names, dir_paths, strict=True))

    elif args.experiments:
        # Convert experiment names to paths
        base_path = Path("src/crackseg/outputs/experiments")
        for exp_name in args.experiments.split(","):
            exp_name = exp_name.strip()
            exp_path = base_path / exp_name
            if exp_path.exists():
                experiments[exp_name] = exp_path

    elif args.auto_find:
        # Find recent experiments automatically
        experiment_dirs = find_experiment_directories()
        for _, exp_dir in enumerate(experiment_dirs[: args.max_experiments]):
            experiments[exp_dir.name] = exp_dir

    else:
        print(
            "❌ No experiments specified. Use --experiments, "
            "--experiment-dirs, or --auto-find"
        )
        return

    if not experiments:
        print("❌ No valid experiments found")
        return

    print(f"🔍 Analyzing {len(experiments)} experiments...")

    # Load data for all experiments
    experiments_data = {}
    for exp_name, exp_path in experiments.items():
        print(f"📂 Loading data for: {exp_name}")
        experiments_data[exp_name] = load_experiment_data(exp_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Training curves
    curves_path = output_dir / f"training_curves_{timestamp}.png"
    plot_training_curves(
        experiments_data,
        title="Training Curves Comparison",
        save_path=curves_path,
    )

    # Performance radar chart
    radar_path = output_dir / f"performance_radar_{timestamp}.png"
    create_performance_radar(
        experiments_data,
        title="Performance Comparison (Radar Chart)",
        save_path=radar_path,
    )

    # Detailed analysis
    comparison_df = create_comparison_table(experiments_data)
    print_detailed_analysis(comparison_df, "EXPERIMENT ANALYSIS")

    # Save comparison table
    csv_path = output_dir / f"experiment_comparison_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"📊 Comparison table saved to: {csv_path}")

    print(f"\n✅ Analysis complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
