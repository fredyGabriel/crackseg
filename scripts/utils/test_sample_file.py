#!/usr/bin/env python3
"""
Test script for running import replacement on a single sample file.

This script tests the replacement process on one file to verify
the changes are correct before applying to all files.

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import os
import sys

# Add the scripts/utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from replace_imports import ImportReplacer


def test_single_file():
    """Test the replacement on a single file."""
    print("Testing import replacement on sample file...")

    # Create replacer with backup enabled
    replacer = ImportReplacer(dry_run=False, backup=True, verbose=True)

    # Test with the file that has the fewest imports
    test_file = "docs/guides/prediction_analysis_guide.md"

    # Override target files to use only our test file
    replacer.target_files = [test_file]

    print(f"Testing replacement on: {test_file}")
    print("Original content preview:")

    # Show original content around the import
    with open(test_file, encoding="utf-8") as f:
        content = f.read()

    # Find the import line
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if (
            "from src.crackseg.evaluation.simple_prediction_analyzer import"
            in line
        ):
            print(f"Line {i + 1}: {line}")
            # Show context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                marker = ">>> " if j == i else "    "
                print(f"{marker}{j + 1:3d}: {lines[j]}")
            break

    # Run the replacement
    success = replacer.run()

    if success:
        print("\n✅ Replacement completed successfully!")
        print("\nModified content preview:")

        # Show modified content
        with open(test_file, encoding="utf-8") as f:
            modified_content = f.read()

        modified_lines = modified_content.split("\n")
        for i, line in enumerate(modified_lines):
            if (
                "from crackseg.evaluation.simple_prediction_analyzer import"
                in line
            ):
                print(f"Line {i + 1}: {line}")
                # Show context
                start = max(0, i - 2)
                end = min(len(modified_lines), i + 3)
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j + 1:3d}: {modified_lines[j]}")
                break

        # Verify backup was created
        backup_file = f"{test_file}.backup"
        if os.path.exists(backup_file):
            print(f"\n✅ Backup created: {backup_file}")
        else:
            print(f"\n❌ Backup not found: {backup_file}")

        return True
    else:
        print("\n❌ Replacement failed!")
        return False


if __name__ == "__main__":
    success = test_single_file()
    sys.exit(0 if success else 1)
