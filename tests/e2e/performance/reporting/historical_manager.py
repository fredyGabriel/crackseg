"""Historical data management module for performance reports.

This module handles loading, storing, and summarizing historical performance
data for trend analysis and comparative reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """Manages historical performance data storage and retrieval."""

    def __init__(self, historical_data_path: Path) -> None:
        """Initialize historical data manager with storage path."""
        self.historical_data_path = historical_data_path
        self.logger = logging.getLogger(__name__)

        # Ensure directory exists
        self.historical_data_path.mkdir(parents=True, exist_ok=True)

    def load_historical_data(self) -> list[dict[str, Any]]:
        """Load historical performance data from stored files."""
        historical_data = []

        # Find all JSON files in the historical data directory
        json_files = list(self.historical_data_path.glob("*.json"))

        if not json_files:
            self.logger.info("No historical data files found")
            return historical_data

        # Sort files by modification time (newest first)
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for file_path in json_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    historical_data.append(data)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load historical data from {file_path}: {e}"
                )

        self.logger.info(
            f"Loaded {len(historical_data)} historical data points"
        )
        return historical_data

    def store_current_data(
        self, processed_data: dict[str, Any], commit_sha: str | None
    ) -> Path:
        """Store current performance data for future historical analysis."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"performance_data_{timestamp}.json"
        file_path = self.historical_data_path / filename

        # Add metadata to stored data
        stored_data = {
            **processed_data,
            "commit_sha": commit_sha,
            "stored_timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stored_data, f, indent=2, default=str)
            self.logger.info(f"Current data stored: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to store current data: {e}")
            raise

    def summarize_historical_data(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate summary of historical data for context."""
        if not historical_data:
            return {
                "data_points": 0,
                "time_range": "no data",
                "avg_success_rate": 0.0,
                "avg_throughput": 0.0,
                "avg_memory_usage": 0.0,
                "avg_violations": 0.0,
            }

        # Extract timestamps for time range calculation
        timestamps = self._extract_timestamps(historical_data)
        time_range = self._calculate_time_range(timestamps)

        # Calculate historical averages
        averages = self._calculate_historical_averages(historical_data)

        return {
            "data_points": len(historical_data),
            "time_range": time_range,
            **averages,
        }

    def _extract_timestamps(
        self, historical_data: list[dict[str, Any]]
    ) -> list[datetime]:
        """Extract valid timestamps from historical data."""
        timestamps = []

        for data_point in historical_data:
            timestamp_str = data_point.get("timestamp", "")
            if timestamp_str:
                try:
                    # Handle both standard ISO format and Z suffix
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str.replace("Z", "+00:00")

                    timestamps.append(datetime.fromisoformat(timestamp_str))
                except ValueError as e:
                    self.logger.debug(
                        f"Invalid timestamp format: {timestamp_str} - {e}"
                    )
                    continue

        return timestamps

    def _calculate_time_range(self, timestamps: list[datetime]) -> str:
        """Calculate time range string from timestamps."""
        if not timestamps:
            return "unknown"

        oldest = min(timestamps)
        newest = max(timestamps)

        return (
            f"{oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}"
        )

    def _calculate_historical_averages(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate average metrics from historical data."""
        success_rates = []
        throughputs = []
        memory_usage = []
        violations = []

        for data_point in historical_data:
            # Extract overall summary metrics
            overall_summary = data_point.get("overall_summary", {})
            success_rate = overall_summary.get("average_success_rate", 0.0)
            throughput = overall_summary.get("average_throughput", 0.0)
            total_violations = overall_summary.get("total_violations", 0)

            if isinstance(success_rate, int | float):
                success_rates.append(success_rate)
            if isinstance(throughput, int | float):
                throughputs.append(throughput)
            if isinstance(total_violations, int | float):
                violations.append(total_violations)

            # Extract resource metrics
            resource_summary = data_point.get("resource_summary", {})
            peak_memory = resource_summary.get("peak_memory_mb", 0.0)
            if isinstance(peak_memory, int | float):
                memory_usage.append(peak_memory)

        return {
            "avg_success_rate": (
                sum(success_rates) / len(success_rates)
                if success_rates
                else 0.0
            ),
            "avg_throughput": (
                sum(throughputs) / len(throughputs) if throughputs else 0.0
            ),
            "avg_memory_usage": (
                sum(memory_usage) / len(memory_usage) if memory_usage else 0.0
            ),
            "avg_violations": (
                sum(violations) / len(violations) if violations else 0.0
            ),
        }

    def cleanup_old_data(self, max_files: int = 100) -> int:
        """Clean up old historical data files, keeping only the most recent."""
        json_files = list(self.historical_data_path.glob("*.json"))

        if len(json_files) <= max_files:
            return 0

        # Sort by modification time (oldest first)
        json_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest files
        files_to_remove = json_files[: len(json_files) - max_files]
        removed_count = 0

        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
                self.logger.debug(f"Removed old data file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {file_path}: {e}")

        if removed_count > 0:
            self.logger.info(
                f"Cleaned up {removed_count} old historical data files"
            )

        return removed_count

    def get_data_file_info(self) -> dict[str, Any]:
        """Get information about stored historical data files."""
        json_files = list(self.historical_data_path.glob("*.json"))

        if not json_files:
            return {
                "total_files": 0,
                "total_size_mb": 0.0,
                "oldest_file": None,
                "newest_file": None,
            }

        # Calculate total size
        total_size = sum(f.stat().st_size for f in json_files)

        # Find oldest and newest files
        json_files.sort(key=lambda f: f.stat().st_mtime)
        oldest_file = json_files[0]
        newest_file = json_files[-1]

        return {
            "total_files": len(json_files),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_file": {
                "name": oldest_file.name,
                "modified": datetime.fromtimestamp(
                    oldest_file.stat().st_mtime, tz=UTC
                ).isoformat(),
            },
            "newest_file": {
                "name": newest_file.name,
                "modified": datetime.fromtimestamp(
                    newest_file.stat().st_mtime, tz=UTC
                ).isoformat(),
            },
        }
