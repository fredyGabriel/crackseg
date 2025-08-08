#!/usr/bin/env python3
"""
Clean up duplicate Hydra output folders.

This script removes the unnecessary '*-run' folders created by Hydra's default
configuration, keeping only the properly named experiment folders.

Safety features:
- Only removes folders ending with '-run'
- Checks if folders contain important data before deletion
- Provides dry-run mode to preview what would be deleted
- Creates backup list of deleted folders

Usage:
    python scripts/maintenance/cleanup_hydra_folders.py        # Dry run (preview)
    python scripts/maintenance/cleanup_hydra_folders.py --execute  # Actually delete
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def find_hydra_run_folders(artifacts_dir: Path) -> list[Path]:
    """Find all folders ending with '-run' in the experiments directory.

    Args:
        artifacts_dir: Path to the artifacts/experiments directory

    Returns:
        List of Path objects for folders ending with '-run'
    """
    run_folders = []

    if not artifacts_dir.exists():
        print(f"Directory not found: {artifacts_dir}")
        return run_folders

    # Find all folders ending with '-run'
    for folder in artifacts_dir.glob("*-run"):
        if folder.is_dir():
            run_folders.append(folder)

    return run_folders


def is_safe_to_delete(folder: Path) -> tuple[bool, str]:
    """Check if a folder is safe to delete.

    A folder is considered safe to delete if:
    - It only contains Hydra configuration files (.hydra/ and *.log)
    - It doesn't contain checkpoints, models, or results

    Args:
        folder: Path to the folder to check

    Returns:
        Tuple of (is_safe, reason)
    """
    # Check for important directories that should not exist in -run folders
    important_dirs = [
        "checkpoints",
        "models",
        "results",
        "metrics",
        "visualizations",
    ]

    for important_dir in important_dirs:
        if (folder / important_dir).exists():
            # Check if directory has actual content
            dir_path = folder / important_dir
            if any(dir_path.iterdir()):
                return False, f"Contains {important_dir}/ with files"

    # Check for important file patterns
    important_patterns = ["*.pth", "*.pt", "*.ckpt", "*.pkl", "*.npy", "*.npz"]

    for pattern in important_patterns:
        if list(folder.rglob(pattern)):
            return False, f"Contains important files matching {pattern}"

    # Check what the folder actually contains
    contents = list(folder.iterdir())

    # Expected contents for a Hydra-only folder
    expected_items = {".hydra", "run.log"}
    actual_items = {item.name for item in contents}

    if actual_items.issubset(expected_items):
        return True, "Only contains Hydra configuration"

    # If it has other content, be cautious
    extra_items = actual_items - expected_items
    if extra_items:
        return False, f"Contains unexpected items: {extra_items}"

    return True, "Safe to delete"


def cleanup_hydra_folders(
    artifacts_dir: Path, execute: bool = False, verbose: bool = True
) -> tuple[list[Path], list[Path]]:
    """Clean up duplicate Hydra folders.

    Args:
        artifacts_dir: Path to artifacts/experiments directory
        execute: If True, actually delete folders. If False, dry run only.
        verbose: If True, print detailed information

    Returns:
        Tuple of (deleted_folders, skipped_folders)
    """
    deleted_folders = []
    skipped_folders = []

    # Find all -run folders
    run_folders = find_hydra_run_folders(artifacts_dir)

    if not run_folders:
        if verbose:
            print("No '*-run' folders found.")
        return deleted_folders, skipped_folders

    if verbose:
        print(f"Found {len(run_folders)} '*-run' folder(s)")
        print("-" * 60)

    # Process each folder
    for folder in sorted(run_folders):
        is_safe, reason = is_safe_to_delete(folder)

        if is_safe:
            if execute:
                try:
                    shutil.rmtree(folder)
                    deleted_folders.append(folder)
                    if verbose:
                        print(f"✅ DELETED: {folder.name}")
                        print(f"   Reason: {reason}")
                except Exception as e:
                    if verbose:
                        print(f"❌ ERROR deleting {folder.name}: {e}")
                    skipped_folders.append(folder)
            else:
                deleted_folders.append(folder)
                if verbose:
                    print(f"🗑️  WOULD DELETE: {folder.name}")
                    print(f"   Reason: {reason}")
        else:
            skipped_folders.append(folder)
            if verbose:
                print(f"⚠️  SKIPPED: {folder.name}")
                print(f"   Reason: {reason}")

        if verbose:
            print()

    return deleted_folders, skipped_folders


def save_cleanup_log(
    deleted_folders: list[Path], skipped_folders: list[Path]
) -> None:
    """Save a log of the cleanup operation.

    Args:
        deleted_folders: List of deleted folder paths
        skipped_folders: List of skipped folder paths
    """
    log_dir = Path("artifacts/logs/cleanup")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hydra_cleanup_{timestamp}.log"

    with open(log_file, "w") as f:
        f.write(f"Hydra Folder Cleanup Log - {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Deleted Folders ({len(deleted_folders)}):\n")
        for folder in deleted_folders:
            f.write(f"  - {folder}\n")

        f.write(f"\nSkipped Folders ({len(skipped_folders)}):\n")
        for folder in skipped_folders:
            f.write(f"  - {folder}\n")

    print(f"\nLog saved to: {log_file}")


def main():
    """Main function to run the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Clean up duplicate Hydra output folders"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete folders (default is dry run)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to artifacts/experiments directory",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Print header
    if not args.quiet:
        print("\n" + "=" * 60)
        print("HYDRA FOLDER CLEANUP UTILITY")
        print("=" * 60)

        if not args.execute:
            print("🔍 DRY RUN MODE - No folders will be deleted")
            print("   Add --execute flag to actually delete folders")
        else:
            print("⚠️  EXECUTE MODE - Folders will be permanently deleted!")

        print("=" * 60 + "\n")

    # Run cleanup
    deleted, skipped = cleanup_hydra_folders(
        args.artifacts_dir, execute=args.execute, verbose=not args.quiet
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        action = "Deleted" if args.execute else "Would delete"
        print(f"{action}: {len(deleted)} folder(s)")
        print(f"Skipped: {len(skipped)} folder(s)")

        if args.execute and deleted:
            save_cleanup_log(deleted, skipped)

        print("=" * 60 + "\n")

    return 0 if not skipped else 1


if __name__ == "__main__":
    exit(main())
