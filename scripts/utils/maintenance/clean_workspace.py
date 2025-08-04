#!/usr/bin/env python
"""
Script to clean up temporary, cache, and obsolete files/folders in the
workspace.

- Removes all __pycache__ and .pytest_cache directories recursively
- Removes all .pyc files recursively
- Removes old log files (crackseg.log)
- Removes artifacts/ folders that are not 'experiments', 'shared', 'global', 'production', 'archive', 'versioning', or 'README.md'
- Removes artifacts/experiment_registry.json if found
- Removes artifacts/checkpoints, artifacts/metrics, artifacts/visualizations if they
  exist outside experiments/
- Removes date-named folders in artifacts/ (e.g., 2025-05-03/)

Usage:
    python scripts/clean_workspace.py
"""

import os
import shutil
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Patterns to clean
CLEAN_DIRS = ["__pycache__", ".pytest_cache"]
CLEAN_FILES = ["*.pyc"]
LOG_FILES = ["crackseg.log"]
ARTIFACTS_KEEP = {
    "experiments",
    "shared",
    "global",
    "production",
    "archive",
    "versioning",
    "README.md",
    "REFACTORING_REPORT.md",
    "UNIFICATION_REPORT.md",
}
DATE_FOLDER_PARTS = 3


def clean_cache_dirs(base: Path):
    """Remove __pycache__ and .pytest_cache directories recursively."""
    for dirpath, _dirnames, _filenames in os.walk(base):
        for d in CLEAN_DIRS:
            full = Path(dirpath) / d
            if full.exists():
                print(f"Removing directory: {full}")
                shutil.rmtree(full, ignore_errors=True)


def clean_pyc_files(base: Path):
    """Remove .pyc files recursively."""
    for pyc in base.rglob("*.pyc"):
        print(f"Removing file: {pyc}")
        pyc.unlink()


def clean_logs(base: Path):
    """Remove log files from the project root."""
    for log in LOG_FILES:
        log_path = base / log
        if log_path.exists():
            print(f"Removing log file: {log_path}")
            log_path.unlink()


def clean_artifacts(artifacts: Path):
    """Clean artifacts/ directory, keeping only allowed folders/files."""
    for item in artifacts.iterdir():
        if item.name not in ARTIFACTS_KEEP:
            print(f"Removing from artifacts/: {item}")
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
    # Remove experiment_registry.json if present
    reg = artifacts / "experiment_registry.json"
    if reg.exists():
        print(f"Removing registry: {reg}")
        reg.unlink()
    # Remove common stray folders
    for stray in ["checkpoints", "metrics", "visualizations"]:
        stray_path = artifacts / stray
        if stray_path.exists():
            print(f"Removing stray folder: {stray_path}")
            shutil.rmtree(stray_path, ignore_errors=True)
    # Remove date-named folders (e.g., 2025-05-03)
    for item in artifacts.iterdir():
        parts = item.name.split("-")
        if (
            item.is_dir()
            and len(parts) == DATE_FOLDER_PARTS
            and all(part.isdigit() for part in parts)
        ):
            print(f"Removing date-named folder: {item}")
            shutil.rmtree(item, ignore_errors=True)


if __name__ == "__main__":
    print("Cleaning workspace...")
    clean_cache_dirs(ROOT)
    clean_pyc_files(ROOT)
    clean_logs(ROOT)
    artifacts_dir = ROOT / "artifacts"
    if artifacts_dir.exists():
        clean_artifacts(artifacts_dir)
    print("Cleanup complete.")
