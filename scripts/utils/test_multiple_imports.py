#!/usr/bin/env python3
"""
Test script for running import replacement on a file with multiple imports.

This script tests the replacement process on a file with multiple
import statements to verify all are replaced correctly.

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import os
import sys

# Add the scripts/utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from replace_imports import ImportReplacer


def test_multiple_imports():
    """Test the replacement on a file with multiple imports."""
    print("Testing import replacement on file with multiple imports...")

    # Create replacer with backup enabled
    replacer = ImportReplacer(dry_run=False, backup=True, verbose=True)

    # Test with the troubleshooting guide that has 2 imports
    test_file = (
        "docs/guides/deployment/deployment_system_troubleshooting_guide.md"
    )

    # Override target files to use only our test file
    replacer.target_files = [test_file]

    print(f"Testing replacement on: {test_file}")
    print("Original content preview:")

    # Show original content around the imports
    with open(test_file, encoding="utf-8") as f:
        content = f.read()

    # Find all import lines
    lines = content.split("\n")
    import_lines = []
    for i, line in enumerate(lines):
        if "from src.crackseg." in line:
            import_lines.append((i, line))

    print(f"Found {len(import_lines)} import statements:")
    for i, (line_num, line) in enumerate(import_lines):
        print(f"  {i + 1}. Line {line_num + 1}: {line}")
        # Show context
        start = max(0, line_num - 1)
        end = min(len(lines), line_num + 2)
        for j in range(start, end):
            marker = ">>> " if j == line_num else "    "
            print(f"{marker}{j + 1:3d}: {lines[j]}")
        print()

    # Run the replacement
    success = replacer.run()

    if success:
        print("\n✅ Replacement completed successfully!")
        print("\nModified content preview:")

        # Show modified content
        with open(test_file, encoding="utf-8") as f:
            modified_content = f.read()

        modified_lines = modified_content.split("\n")
        new_import_lines = []
        for i, line in enumerate(modified_lines):
            if "from crackseg." in line:
                new_import_lines.append((i, line))

        print(f"Found {len(new_import_lines)} updated import statements:")
        for i, (line_num, line) in enumerate(new_import_lines):
            print(f"  {i + 1}. Line {line_num + 1}: {line}")
            # Show context
            start = max(0, line_num - 1)
            end = min(len(modified_lines), line_num + 2)
            for j in range(start, end):
                marker = ">>> " if j == line_num else "    "
                print(f"{marker}{j + 1:3d}: {modified_lines[j]}")
            print()

        # Verify backup was created
        backup_file = f"{test_file}.backup"
        if os.path.exists(backup_file):
            print(f"\n✅ Backup created: {backup_file}")
        else:
            print(f"\n❌ Backup not found: {backup_file}")

        # Verify no old imports remain
        old_imports = [
            line for line in modified_lines if "from src.crackseg." in line
        ]
        if old_imports:
            print(
                f"\n❌ Found {len(old_imports)} old imports that weren't replaced:"
            )
            for line in old_imports:
                print(f"  {line}")
        else:
            print("\n✅ All old imports were successfully replaced!")

        return True
    else:
        print("\n❌ Replacement failed!")
        return False


if __name__ == "__main__":
    success = test_multiple_imports()
    sys.exit(0 if success else 1)
