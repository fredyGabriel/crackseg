"""Utilities for loading training data artifacts for visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_training_data(
    experiment_dir: Path, include_gradients: bool = False
) -> dict[str, Any]:
    """Load metrics, summary, config, and optional gradients from an experiment.

    Mirrors legacy behavior used by advanced training visualizer.
    """
    training_data: dict[str, Any] = {}

    # Load metrics data
    metrics_file = experiment_dir / "metrics" / "metrics.jsonl"
    if metrics_file.exists():
        metrics_data: list[dict[str, Any]] = []
        with open(metrics_file, encoding="utf-8") as f:
            for line in f:
                metrics_data.append(json.loads(line.strip()))
        training_data["metrics"] = metrics_data

    # Load summary data
    summary_file = experiment_dir / "metrics" / "complete_summary.json"
    if summary_file.exists():
        with open(summary_file, encoding="utf-8") as f:
            training_data["summary"] = json.load(f)

    # Load configuration (kept as JSON for backward-compatibility)
    config_file = experiment_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            training_data["config"] = json.load(f)

    # Load gradient data if requested
    if include_gradients:
        gradient_file = experiment_dir / "metrics" / "gradients.jsonl"
        if gradient_file.exists():
            gradient_data: list[dict[str, Any]] = []
            with open(gradient_file, encoding="utf-8") as f:
                for line in f:
                    gradient_data.append(json.loads(line.strip()))
            training_data["gradients"] = gradient_data

    return training_data
