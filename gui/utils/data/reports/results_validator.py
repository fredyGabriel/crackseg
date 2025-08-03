"""
Results validation utilities for GUI components. This module provides
validation functions for training results, metrics files, and run
directories to ensure data integrity and proper format.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ResultsValidator:
    """
    Validator for training results and metrics files. Provides methods to
    validate run directories, metrics files, and ensure data integrity for
    the GUI results display components.
    """

    def __init__(self) -> None:
        """Initialize the results validator."""
        self.required_metrics = ["loss", "iou", "dice", "precision", "recall"]
        self.optional_metrics = [
            "f1_score",
            "boundary_f1",
            "hausdorff_distance",
            "average_precision",
        ]

    def validate_run_directory(self, run_dir: str) -> bool:
        """
        Validate a training run directory structure. Args: run_dir: Path to
        the run directory to validate. Returns: True if the directory is
        valid, False otherwise.
        """
        try:
            run_path = Path(run_dir)
            if not run_path.exists() or not run_path.is_dir():
                return False

            # Check for required files
            required_files = ["config.yaml", "metrics.json"]
            for file_name in required_files:
                if not (run_path / file_name).exists():
                    return False

            # Validate metrics file
            metrics_file = run_path / "metrics.json"
            return self.validate_metrics_file(str(metrics_file))

        except (OSError, ValueError):
            return False

    def validate_metrics_file(self, metrics_path: str) -> bool:
        """
        Validate a metrics JSON file. Args: metrics_path: Path to the metrics
        file to validate. Returns: True if the metrics file is valid, False
        otherwise.
        """
        try:
            metrics_file = Path(metrics_path)
            if not metrics_file.exists():
                return False

            with open(metrics_file, encoding="utf-8") as f:
                metrics_data: dict[str, Any] = json.load(f)

            # Check for required metrics
            for metric_name in self.required_metrics:
                if metric_name not in metrics_data:
                    return False

            # Validate metric values
            for value in metrics_data.values():
                if not self._is_valid_metric_value(value):
                    return False

            return True

        except (OSError, json.JSONDecodeError, ValueError):
            return False

    def validate_csv_metrics(self, csv_path: str) -> bool:
        """
        Validate a CSV metrics file. Args: csv_path: Path to the CSV metrics
        file to validate. Returns: True if the CSV file is valid, False
        otherwise.
        """
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                return False

            # Try to read CSV
            df = pd.read_csv(csv_file)  # type: ignore[misc]
            if df.empty:
                return False

            # Check for required columns
            required_columns = ["epoch"] + self.required_metrics
            for col in required_columns:
                if col not in df.columns:
                    return False

            # Validate numeric columns
            for metric in self.required_metrics:
                if not pd.api.types.is_numeric_dtype(df[metric]):  # type: ignore[misc]
                    return False

            return True

        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return False

    def _is_valid_metric_value(self, value: Any) -> bool:
        """
        Check if a metric value is valid. Args: value: The metric value to
        validate. Returns: True if the value is valid, False otherwise.
        """
        # Check if it's a numeric value
        if not isinstance(value, int | float):
            return False

        # Check if it's finite (not inf or NaN) - Fix logical error
        if isinstance(value, float) and (
            value == float("inf") or value == float("-inf") or value != value
        ):
            return False

        # Most metrics should be in [0, 1] range
        if 0 <= value <= 1:
            return True

        # Loss values can be any positive number
        if value >= 0:
            return True

        return False

    def get_validation_errors(self, run_dir: str) -> list[str]:
        """
        Get detailed validation errors for a run directory. Args: run_dir:
        Path to the run directory to validate. Returns: List of validation
        error messages.
        """
        errors: list[str] = []
        run_path = Path(run_dir)

        if not run_path.exists():
            errors.append(f"Run directory does not exist: {run_dir}")
            return errors

        if not run_path.is_dir():
            errors.append(f"Path is not a directory: {run_dir}")
            return errors

        # Check required files
        required_files = ["config.yaml", "metrics.json"]
        for file_name in required_files:
            file_path = run_path / file_name
            if not file_path.exists():
                errors.append(f"Missing required file: {file_name}")

        # Check metrics file
        metrics_file = run_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    metrics_data = json.load(f)

                if not isinstance(metrics_data, dict):
                    errors.append("Metrics file is not a valid JSON object")
                else:
                    for metric_name in self.required_metrics:
                        if metric_name not in metrics_data:
                            errors.append(
                                f"Missing required metric: {metric_name}"
                            )

            except (OSError, json.JSONDecodeError) as e:
                errors.append(f"Invalid metrics file: {e}")

        return errors
