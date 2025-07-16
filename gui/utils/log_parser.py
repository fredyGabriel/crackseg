"""
Log Parsing Utilities

This module provides functions to parse real-time logs from the training
process and extract structured data, such as metrics.
"""

import re
from typing import Any

import pandas as pd


def parse_metrics_from_log_line(line: str) -> dict[str, float] | None:
    """
    Parses a log line to extract training or validation metrics.

    Args:
        line: A single line of log output.

    Returns:
        A dictionary of metrics if found, otherwise None.
    """
    # Regex for validation logs, e.g., "Validation - Epoch: [1/10],
    # Val Loss: 0.123, F1: 0.987"
    val_pattern = re.compile(
        r".*Epoch:\s*\[(\d+)/\d+\].*Val Loss:\s*([\d.]+).*"
    )
    val_match = val_pattern.match(line)
    if val_match:
        epoch = int(val_match.group(1))
        val_loss = float(val_match.group(2))
        metrics = {"epoch": epoch, "val_loss": val_loss}

        # Dynamically find other metrics in the same line
        additional_metrics = re.findall(r"(\w+):\s*([\d.]+)", line)
        for name, value in additional_metrics:
            if name.lower() not in ["epoch", "val loss"]:
                # Sanitize name for consistency
                metric_name = f"val_{name.lower()}"
                metrics[metric_name] = float(value)
        return metrics

    # Regex for training loss, e.g., "Epoch [1/10], Batch [50/100],
    # Train Loss: 0.456"
    train_pattern = re.compile(
        r".*Epoch\s*\[(\d+)/\d+\].*Train Loss:\s*([\d.]+)"
    )
    train_match = train_pattern.match(line)
    if train_match:
        return {
            "epoch": int(train_match.group(1)),
            "train_loss": float(train_match.group(2)),
        }

    return None


def initialize_metrics_df() -> pd.DataFrame:
    """Creates an empty DataFrame to store metrics."""
    # Create empty DataFrame with explicit structure to avoid type issues
    data: dict[str, Any] = {"epoch": [], "train_loss": [], "val_loss": []}
    df = pd.DataFrame(data)
    df = df.set_index("epoch")
    return df


def update_metrics_df(
    df: pd.DataFrame, metrics: dict[str, float]
) -> pd.DataFrame:
    """
    Updates the metrics DataFrame with a new dictionary of metrics.

    Args:
        df: The existing pandas DataFrame of metrics.
        metrics: A new dictionary of metrics to add or update.

    Returns:
        The updated DataFrame.
    """
    epoch = metrics.pop("epoch")
    for col, value in metrics.items():
        # Add column if it doesn't exist
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[epoch, col] = value
    return df
