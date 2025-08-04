"""Core experiment visualization functionality.

This module provides core functionality for experiment visualization
including data loading and basic experiment analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentVisualizer:
    """Create visualizations for experiment results and comparisons."""

    def __init__(self) -> None:
        """Initialize the experiment visualizer."""

    def load_experiment_data(self, experiment_dir: Path) -> dict[str, Any]:
        """Load all data for an experiment including metrics and logs.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary containing experiment data
        """
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

    def create_comparison_table(
        self, experiments_data: dict[str, dict]
    ) -> pd.DataFrame:
        """Create a comparison table of all experiments.

        Args:
            experiments_data: Dictionary mapping experiment names to data

        Returns:
            DataFrame with experiment comparisons
        """
        rows = []

        for exp_name, data in experiments_data.items():
            if "summary" in data:
                summary = data["summary"]
                best_metrics = summary.get("best_metrics", {})

                row = {
                    "Experiment": exp_name,
                    "Final Loss": best_metrics.get("loss", {}).get(
                        "value", None
                    ),
                    "Final IoU": best_metrics.get("iou", {}).get(
                        "value", None
                    ),
                    "Final F1": best_metrics.get("f1", {}).get("value", None),
                    "Final Precision": best_metrics.get("precision", {}).get(
                        "value", None
                    ),
                    "Final Recall": best_metrics.get("recall", {}).get(
                        "value", None
                    ),
                    "Total Epochs": summary.get("experiment_info", {}).get(
                        "total_epochs", None
                    ),
                    "Best Epoch": summary.get("experiment_info", {}).get(
                        "best_epoch", None
                    ),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def find_experiment_directories(
        self, base_path: str = "artifacts/experiments"
    ) -> list[Path]:
        """Find all experiment directories.

        Args:
            base_path: Base path to search for experiments

        Returns:
            List of experiment directory paths
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return []

        experiment_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and (item / "metrics").exists():
                experiment_dirs.append(item)

        return experiment_dirs
