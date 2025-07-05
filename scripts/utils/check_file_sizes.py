#!/usr/bin/env python3
"""File size monitoring utility for CrackSeg project.

This script checks all Python files for line count violations and suggests
refactoring candidates based on the project's 300-line preferred limit
and 400-line absolute maximum.
"""

import argparse
from pathlib import Path
from typing import NamedTuple


class FileSizeInfo(NamedTuple):
    """File size information."""

    path: Path
    lines: int
    status: str  # 'ok', 'warning', 'violation'


def count_lines(file_path: Path) -> int:
    """Count lines in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except (OSError, UnicodeDecodeError):
        return 0


def analyze_file(file_path: Path) -> FileSizeInfo:
    """Analyze a single file for size violations."""
    lines = count_lines(file_path)

    if lines <= 300:
        status = "ok"
    elif lines <= 400:
        status = "warning"
    else:
        status = "violation"

    return FileSizeInfo(file_path, lines, status)


def scan_project() -> list[FileSizeInfo]:
    """Scan all Python files in the project."""
    project_root = Path(__file__).parent.parent.parent

    directories = ["src", "tests", "scripts"]
    files = []

    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            files.extend(dir_path.rglob("*.py"))

    return [analyze_file(f) for f in files]


def print_report(files: list[FileSizeInfo]) -> None:
    """Print a detailed size report."""
    violations = [f for f in files if f.status == "violation"]
    warnings = [f for f in files if f.status == "warning"]
    ok_files = [f for f in files if f.status == "ok"]

    print("🚨 FILE SIZE ANALYSIS REPORT")
    print("=" * 50)

    if violations:
        print(f"\n❌ CRITICAL VIOLATIONS (>{400} lines):")
        for file_info in sorted(
            violations, key=lambda x: x.lines, reverse=True
        ):
            relative_path = file_info.path.relative_to(Path.cwd())
            print(f"  {file_info.lines:4d} lines: {relative_path}")

    if warnings:
        print("\n⚠️  SIZE WARNINGS (300-400 lines):")
        for file_info in sorted(warnings, key=lambda x: x.lines, reverse=True):
            relative_path = file_info.path.relative_to(Path.cwd())
            print(f"  {file_info.lines:4d} lines: {relative_path}")

    print("\n✅ SUMMARY:")
    print(f"  Total files analyzed: {len(files)}")
    print(f"  Within limits (≤300): {len(ok_files)}")
    print(f"  Size warnings (301-400): {len(warnings)}")
    print(f"  Critical violations (>400): {len(violations)}")

    if violations:
        print("\n🎯 REFACTORING PRIORITY:")
        print("  Top 3 files need immediate attention:")
        for i, file_info in enumerate(
            sorted(violations, key=lambda x: x.lines, reverse=True)[:3], 1
        ):
            relative_path = file_info.path.relative_to(Path.cwd())
            ratio = file_info.lines / 400
            print(f"    {i}. {relative_path} ({ratio:.1f}x over limit)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check file sizes in CrackSeg project"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all files, not just violations",
    )

    args = parser.parse_args()

    files = scan_project()

    if args.verbose:
        print("All files:")
        for file_info in sorted(files, key=lambda x: x.lines, reverse=True):
            relative_path = file_info.path.relative_to(Path.cwd())
            status_icon = {"ok": "✅", "warning": "⚠️", "violation": "❌"}[
                file_info.status
            ]
            print(
                f"  {status_icon} {file_info.lines:4d} lines: {relative_path}"
            )
        print()

    print_report(files)


if __name__ == "__main__":
    main()
