"""
Git metadata collection for ExperimentTracker.

This module provides Git metadata collection methods for the ExperimentTracker
component.
"""

import subprocess
from pathlib import Path

from crackseg.utils.experiment.metadata import ExperimentMetadata


class ExperimentGitManager:
    """Manages Git metadata collection for experiments."""

    @staticmethod
    def collect_git_metadata(metadata: ExperimentMetadata) -> None:
        """Collect Git metadata for the experiment."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                metadata.git_commit = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                metadata.git_branch = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True,
                cwd=Path.cwd(),
            )
            metadata.git_dirty = result.returncode != 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
