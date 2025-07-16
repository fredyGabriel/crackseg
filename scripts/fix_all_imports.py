#!/usr/bin/env python3
"""
Comprehensive import fixer for the crackseg project.
Fixes all remaining 'from crackseg.' imports to 'from crackseg.' imports.
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path: Path) -> tuple[bool, int]:
    """
    Fix imports in a single file.

    Returns:
        (was_modified, number_of_changes)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Pattern 1: from crackseg. -> from crackseg.
        content, count1 = re.subn(r"from src\.", "from crackseg.", content)

        # Pattern 2: import crackseg. -> import crackseg.
        content, count2 = re.subn(r"import src\.", "import crackseg.", content)

        total_changes = count1 + count2

        if total_changes > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed {total_changes} import(s) in: {file_path}")
            return True, total_changes

        return False, 0

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False, 0


def main():
    """Main function to fix all imports in the project."""
    print("🔍 Searching for files with incorrect imports...")

    # Define directories to search
    search_dirs = ["src", "tests", "scripts", "gui"]

    # Also search root-level Python files
    root_files = ["run.py", "main.py", "__main__.py"]

    total_files_modified = 0
    total_changes = 0

    # Search in directories
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, _dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = Path(root) / file
                        was_modified, changes = fix_imports_in_file(file_path)
                        if was_modified:
                            total_files_modified += 1
                            total_changes += changes

    # Search root-level files
    for file_name in root_files:
        file_path = Path(file_name)
        if file_path.exists():
            was_modified, changes = fix_imports_in_file(file_path)
            if was_modified:
                total_files_modified += 1
                total_changes += changes

    print("\n🎉 Completed!")
    print(f"📊 Files modified: {total_files_modified}")
    print(f"📊 Total changes made: {total_changes}")

    if total_changes == 0:
        print("✨ No incorrect imports found - everything looks good!")


if __name__ == "__main__":
    main()
