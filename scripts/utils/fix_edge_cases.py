#!/usr/bin/env python3
"""
Edge Case Fixer for Import Replacement

This script handles edge cases that were not caught by the main replacement script.
Specifically targets:
1. logging.getLogger("src.crackseg...") patterns
2. python -m src.crackseg... command patterns
3. Other non-import references to src.crackseg

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


class EdgeCaseFixer:
    """Handles edge cases in import replacement that require special attention."""

    def __init__(
        self, dry_run: bool = True, backup: bool = True, verbose: bool = False
    ):
        self.dry_run = dry_run
        self.backup = backup
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "total_replacements": 0,
            "errors": 0,
        }

        # Edge case patterns
        self.edge_patterns = [
            # logging.getLogger patterns
            (
                r'logging\.getLogger\("src\.crackseg\.',
                r'logging.getLogger("crackseg.',
            ),
            # python -m src.crackseg patterns
            (r"python -m src\.crackseg\.", r"python -m crackseg."),
            # python -m src.evaluation patterns
            (r"python -m src\.evaluation", r"python -m crackseg.evaluation"),
            # Other src.crackseg references in code blocks
            (r"`src\.crackseg\.", r"`crackseg."),
        ]

    def process_file(self, file_path: str) -> bool:
        """Process a single file for edge cases."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                self.stats["errors"] += 1
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            replacements_made = 0

            # Apply each edge case pattern
            for pattern, replacement in self.edge_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    if self.verbose:
                        logger.info(
                            f"  Found {len(matches)} matches for pattern: {pattern}"
                        )

                    content = re.sub(pattern, replacement, content)
                    replacements_made += len(matches)

            # Check if any changes were made
            if content != original_content:
                if self.backup and not self.dry_run:
                    backup_path = f"{file_path}.edge_backup"
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"  Created backup: {backup_path}")

                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(
                        f"  Modified {file_path} ({replacements_made} replacements)"
                    )
                else:
                    logger.info(
                        f"  Would modify {file_path} ({replacements_made} replacements)"
                    )

                self.stats["files_modified"] += 1
                self.stats["total_replacements"] += replacements_made
            else:
                if self.verbose:
                    logger.info(f"  No changes needed in {file_path}")

            self.stats["files_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats["errors"] += 1
            return False

    def process_directory(self, directory: str) -> None:
        """Process all markdown files in a directory for edge cases."""
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
        """Print processing summary."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("EDGE CASE FIXING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Total replacements: {self.stats['total_replacements']}")
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
    """Main function for edge case fixing."""
    parser = argparse.ArgumentParser(
        description="Fix edge cases in import replacement"
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
    fixer = EdgeCaseFixer(
        dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose
    )

    # Process directory
    fixer.process_directory(args.directory)


if __name__ == "__main__":
    main()
