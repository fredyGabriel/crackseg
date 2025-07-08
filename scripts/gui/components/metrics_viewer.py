"""
Metrics viewer component stub for the CrackSeg application.

This module provides a temporary stub implementation of the metrics viewer
component while the full implementation is being developed.
"""

from typing import Any

import pandas as pd


def create_metrics_placeholder() -> None:
    """Create a placeholder for metrics display.

    This is a stub implementation that does nothing.
    In the future, this would create placeholders for real-time metrics.
    """
    pass


def update_metrics_df(
    metrics_df: pd.DataFrame, new_data: Any = None
) -> pd.DataFrame:
    """Update the metrics dataframe with new data.

    Args:
        metrics_df: Current metrics dataframe.
        new_data: New data to add (currently unused in stub).

    Returns:
        The unmodified dataframe (stub behavior).
    """
    return metrics_df
