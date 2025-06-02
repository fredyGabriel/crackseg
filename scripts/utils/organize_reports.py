#!/usr/bin/env python3
"""
Script to automatically organize CrackSeg project reports.

This script maintains the established organizational structure of reports,
moving scattered files to their correct locations according to conventions.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path


class ReportOrganizer:
    """Automatic organizer for project reports."""

    def __init__(self, project_root: str) -> None:
        """Initialize the organizer with the project root."""
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "docs" / "reports"

        # Define file patterns and their destinations
        self.file_patterns: dict[str, str] = {
            # Testing reports
            r".*test.*priority.*\.md$": "testing",
            r".*test.*improvement.*plan.*\.md$": "testing",
            r".*test.*inventory.*\.txt$": "testing",
            r".*test.*pattern.*\.md$": "testing",
            # Coverage reports
            r".*coverage.*comparison.*\.md$": "coverage",
            r".*coverage.*gap.*\.md$": "coverage",
            r".*coverage.*analysis.*\.md$": "coverage",
            r".*coverage.*validation.*\.md$": "coverage",
            # Task reports
            r".*task.*completion.*\.md$": "tasks",
            r".*task.*complexity.*\.json$": "tasks",
            r"temp_update.*\.txt$": "tasks",
            # Model reports
            r".*model.*import.*\.json$": "models",
            r".*model.*inventory.*\.json$": "models",
            r".*model.*structure.*\.json$": "models",
            r".*model.*expected.*\.json$": "models",
            r".*model.*pyfiles.*\.json$": "models",
            # Project reports
            r".*plan.*verificacion.*\.md$": "project",
            r".*project.*plan.*\.md$": "project",
            # Archive (old reports with timestamps)
            r".*stats.*report.*\d{8}.*\.txt$": "archive",
            r".*report.*\d{8}_\d{6}.*\.(txt|md)$": "archive",
        }

    def scan_for_reports(self) -> list[tuple[Path, str]]:
        """Scan the project for scattered reports."""
        reports_found: list[tuple[Path, str]] = []

        # Directories to scan
        scan_dirs = [
            self.project_root,  # Root
            self.project_root / "outputs",
            self.project_root / "scripts" / "reports",
            self.project_root / "scripts",
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            for file_path in scan_dir.rglob("*"):
                if file_path.is_file():
                    # Check if file is already in correct location
                    if self.reports_dir in file_path.parents:
                        continue

                    # Search for matching pattern
                    for pattern, destination in self.file_patterns.items():
                        if re.match(pattern, file_path.name, re.IGNORECASE):
                            reports_found.append((file_path, destination))
                            break

        return reports_found

    def organize_reports(self, dry_run: bool = False) -> dict[str, list[str]]:
        """Organize found reports."""
        reports = self.scan_for_reports()
        results: dict[str, list[str]] = {
            "moved": [],
            "errors": [],
            "skipped": [],
        }

        for file_path, destination in reports:
            try:
                dest_dir = self.reports_dir / destination
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / file_path.name

                # Check if file already exists in destination
                if dest_path.exists():
                    results["skipped"].append(
                        f"{file_path} -> {dest_path} (already exists)"
                    )
                    continue

                if not dry_run:
                    shutil.move(str(file_path), str(dest_path))

                results["moved"].append(f"{file_path} -> {dest_path}")

            except Exception as e:
                results["errors"].append(f"Error moving {file_path}: {e}")

        return results

    def clean_empty_dirs(self) -> list[str]:
        """Clean empty directories after moving files."""
        cleaned: list[str] = []

        # Directories that may become empty
        check_dirs = [
            self.project_root / "outputs",
            self.project_root / "scripts" / "reports",
        ]

        for check_dir in check_dirs:
            if check_dir.exists() and check_dir.is_dir():
                try:
                    # Try to remove if empty (except README.md)
                    contents = list(check_dir.iterdir())
                    if not contents or (
                        len(contents) == 1 and contents[0].name == "README.md"
                    ):
                        # Don't remove, just report
                        cleaned.append(f"Nearly empty directory: {check_dir}")
                except Exception:
                    pass

        return cleaned

    def generate_report(self) -> str:
        """Generate current organization report."""
        report_lines = [
            "# Report Organization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Current Structure:",
            "",
        ]

        for category_dir in self.reports_dir.iterdir():
            if category_dir.is_dir() and category_dir.name != "__pycache__":
                report_lines.append(f"### {category_dir.name.title()}/")
                files = list(category_dir.glob("*"))
                if files:
                    for file_path in sorted(files):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            size_str = (
                                f"{size:,} bytes"
                                if size < 1024
                                else f"{size / 1024:.1f} KB"
                            )
                            report_lines.append(
                                f"- `{file_path.name}` ({size_str})"
                            )
                else:
                    report_lines.append("- (empty)")
                report_lines.append("")

        return "\n".join(report_lines)


def main() -> None:
    """Main function of the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize CrackSeg project reports"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only show what would be moved"
    )
    parser.add_argument(
        "--project-root", default=".", help="Project root directory"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate current structure report",
    )

    args = parser.parse_args()

    organizer = ReportOrganizer(args.project_root)

    if args.report:
        print(organizer.generate_report())
        return

    print("🔍 Scanning for scattered reports...")
    results = organizer.organize_reports(dry_run=args.dry_run)

    if args.dry_run:
        print("\n📋 SIMULATION - Files that would be moved:")
    else:
        print("\n✅ Files organized:")

    for moved in results["moved"]:
        print(f"  📁 {moved}")

    if results["skipped"]:
        print("\n⏭️ Files skipped:")
        for skipped in results["skipped"]:
            print(f"  ⚠️ {skipped}")

    if results["errors"]:
        print("\n❌ Errors:")
        for error in results["errors"]:
            print(f"  🚨 {error}")

    if not args.dry_run:
        cleaned = organizer.clean_empty_dirs()
        if cleaned:
            print("\n🧹 Directories reviewed:")
            for clean in cleaned:
                print(f"  📂 {clean}")

    # Split long line to comply with E501
    moved_count = len(results["moved"])
    skipped_count = len(results["skipped"])
    error_count = len(results["errors"])
    print(
        f"\n📊 Summary: {moved_count} moved, {skipped_count} skipped, "
        f"{error_count} errors"
    )


if __name__ == "__main__":
    main()
