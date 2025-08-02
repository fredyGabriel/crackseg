#!/usr/bin/env python3
"""
Code Snippet Fixer

This script fixes common syntax errors in code snippets in documentation files.
It handles issues like incomplete code blocks, missing colons, and malformed imports.

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import argparse
import logging
import os
import re
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeSnippetFixer:
    """Fixes common syntax errors in code snippets."""

    def __init__(
        self, dry_run: bool = True, backup: bool = True, verbose: bool = False
    ):
        self.dry_run = dry_run
        self.backup = backup
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "snippets_fixed": 0,
            "errors": 0,
        }

    def _fix_incomplete_block(self, content: str) -> str:
        """Fix incomplete code blocks."""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Fix class definitions
            if re.match(r"class\s+\w+\s*$", line):
                fixed_lines.append(line + ":")
                fixed_lines.append("    pass")
            # Fix function definitions
            elif re.match(r"def\s+\w+\s*\([^)]*\)\s*$", line):
                fixed_lines.append(line + ":")
                fixed_lines.append("    pass")
            # Fix incomplete imports
            elif re.match(
                r"from crackseg\.[\w\.]+\s+import\s+[\w\s,]+$", line
            ):
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def process_file(self, file_path: str) -> bool:
        """Process a single file for code snippet fixes."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                self.stats["errors"] += 1
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            fixes_made = 0

            # Apply fixes
            # Remove invalid characters in code blocks
            if "❌" in content:
                content = content.replace("❌", "")
                fixes_made += 1
                if self.verbose:
                    logger.info("  Applied fix: Remove invalid characters")

            # Fix incomplete imports
            import_pattern = r"from crackseg\.[\w\.]+\s+import\s+[\w\s,]+$"
            if re.search(import_pattern, content, re.MULTILINE):
                content = re.sub(
                    import_pattern,
                    lambda m: m.group(0) + "\n",
                    content,
                    flags=re.MULTILINE,
                )
                fixes_made += 1
                if self.verbose:
                    logger.info("  Applied fix: Fix incomplete imports")

            # Fix class definitions without colons
            class_pattern = r"class\s+\w+\s*$"
            if re.search(class_pattern, content, re.MULTILINE):
                content = re.sub(
                    class_pattern,
                    lambda m: m.group(0) + ":\n    pass",
                    content,
                    flags=re.MULTILINE,
                )
                fixes_made += 1
                if self.verbose:
                    logger.info("  Applied fix: Fix class definitions")

            # Fix function definitions without colons
            func_pattern = r"def\s+\w+\s*\([^)]*\)\s*$"
            if re.search(func_pattern, content, re.MULTILINE):
                content = re.sub(
                    func_pattern,
                    lambda m: m.group(0) + ":\n    pass",
                    content,
                    flags=re.MULTILINE,
                )
                fixes_made += 1
                if self.verbose:
                    logger.info("  Applied fix: Fix function definitions")

            # Check if any changes were made
            if content != original_content:
                if self.backup and not self.dry_run:
                    backup_path = f"{file_path}.snippet_backup"
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"  Created backup: {backup_path}")

                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"  Modified {file_path} ({fixes_made} fixes)")
                else:
                    logger.info(
                        f"  Would modify {file_path} ({fixes_made} fixes)"
                    )

                self.stats["files_modified"] += 1
                self.stats["snippets_fixed"] += fixes_made
            else:
                if self.verbose:
                    logger.info(f"  No fixes needed in {file_path}")

            self.stats["files_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats["errors"] += 1
            return False

    def process_directory(self, directory: str) -> None:
        """Process all markdown files in a directory."""
        logger.info(f"Processing directory: {directory}")

        # Find all markdown files
        md_files = []
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        logger.info(f"Found {len(md_files)} markdown files")

        # Process each file
        for file_path in md_files:
            if self.verbose:
                logger.info(f"Processing: {file_path}")
            self.process_file(file_path)

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print fixing summary."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("CODE SNIPPET FIXING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Snippets fixed: {self.stats['snippets_fixed']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.dry_run:
            logger.info("Mode: DRY RUN (no changes made)")
        else:
            logger.info("Mode: LIVE (changes applied)")

        if self.backup:
            logger.info("Backup: Enabled")
        else:
            logger.info("Backup: Disabled")


def main():
    """Main function for code snippet fixing."""
    parser = argparse.ArgumentParser(
        description="Fix common syntax errors in code snippets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backup files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--directory",
        default="docs",
        help="Directory to process (default: docs)",
    )

    args = parser.parse_args()

    # Initialize fixer
    fixer = CodeSnippetFixer(
        dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose
    )

    # Process directory
    fixer.process_directory(args.directory)


if __name__ == "__main__":
    main()
