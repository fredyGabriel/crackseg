#!/usr/bin/env python3
"""
Test File Coverage Checker - CrackSeg Project

This script ensures that newly added source files have corresponding test
files, helping maintain test coverage discipline during development.

Usage:
    python scripts/check_test_files.py [source_files...]
"""

import sys
from pathlib import Path


def check_test_files(source_files: list[str]) -> tuple[bool, list[str]]:
    """
    Check if source files have corresponding test files.

    Args:
        source_files: List of source file paths

    Returns:
        Tuple of (all_have_tests, missing_test_files)
    """
    missing_tests = []

    for source_file in source_files:
        source_path = Path(source_file)

        # Skip __init__.py files and non-Python files
        if source_path.name == "__init__.py" or source_path.suffix != ".py":
            continue

        # Skip if file doesn't exist (could be a deleted file in git)
        if not source_path.exists():
            continue

        # Convert src path to corresponding test path
        if source_path.parts[0] == "src":
            # Remove 'src' and add 'tests/unit' prefix
            test_parts = ["tests", "unit"] + list(source_path.parts[1:])
            test_path = Path(*test_parts)

            # Change filename to test_*.py format
            test_filename = f"test_{test_path.stem}.py"
            test_path = test_path.parent / test_filename

            # Check if test file exists
            if not test_path.exists():
                missing_tests.append(f"{source_file} -> {test_path}")

    return len(missing_tests) == 0, missing_tests


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("No source files provided to check.")
        return 0

    source_files = sys.argv[1:]
    all_have_tests, missing_tests = check_test_files(source_files)

    if not all_have_tests:
        print("❌ Missing test files for the following source files:")
        for missing in missing_tests:
            print(f"  - {missing}")
        print("\nPlease create corresponding test files before committing.")
        print(
            "Test files should be located in tests/unit/ with 'test_' prefix."
        )
        return 1

    print("✅ All source files have corresponding test files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
