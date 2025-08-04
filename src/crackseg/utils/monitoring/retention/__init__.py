"""Data retention system for CrackSeg project.

This module provides configurable data retention policies to manage
memory usage and storage of monitoring metrics over time.
"""

from .policies import (
    CompositeRetentionPolicy,
    CountBasedRetentionPolicy,
    RetentionManager,
    RetentionPolicy,
    TimeBasedRetentionPolicy,
)

__all__ = [
    "RetentionPolicy",
    "TimeBasedRetentionPolicy",
    "CountBasedRetentionPolicy",
    "CompositeRetentionPolicy",
    "RetentionManager",
]
