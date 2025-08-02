#!/usr/bin/env python3
"""
Automated Import Replacement Script

This script replaces 'from src.' import statements with 'from crackseg.'
in documentation files as part of the documentation update process.

Usage:
    python scripts/utils/replace_imports.py [--dry-run] [--backup] [--verbose]

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImportReplacer:
    """Handles automated replacement of import statements in documentation files."""

    def __init__(
        self, dry_run: bool = False, backup: bool = True, verbose: bool = False
    ):
        """
        Initialize the import replacer.

        Args:
            dry_run: If True, show what would be changed without making changes
            backup: If True, create backup files before making changes
            verbose: If True, show detailed output
        """
        self.dry_run = dry_run
        self.backup = backup
        self.verbose = verbose

        # Import patterns to replace
        self.import_patterns = [
            (
                r"from src\.crackseg\.utils\.deployment\.config import",
                "from crackseg.utils.deployment.config import",
            ),
            (
                r"from src\.crackseg\.utils\.deployment\.orchestration import",
                "from crackseg.utils.deployment.orchestration import",
            ),
            (
                r"from src\.crackseg\.utils\.deployment\.health_monitoring import",
                "from crackseg.utils.deployment.health_monitoring import",
            ),
            (
                r"from src\.crackseg\.utils\.deployment\.multi_target import",
                "from crackseg.utils.deployment.multi_target import",
            ),
            (
                r"from src\.crackseg\.evaluation\.simple_prediction_analyzer import",
                "from crackseg.evaluation.simple_prediction_analyzer import",
            ),
        ]

        # Files identified in the scan report
        self.target_files = [
            "docs/guides/health_monitoring_guide.md",
            "docs/guides/deployment/deployment_system_configuration_guide.md",
            "docs/guides/deployment/deployment_system_user_guide.md",
            "docs/guides/multi_target_deployment_guide.md",
            "docs/guides/prediction_analysis_guide.md",
            "docs/guides/deployment/deployment_system_troubleshooting_guide.md",
        ]

        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "total_replacements": 0,
            "errors": 0,
        }

    def validate_file_exists(self, file_path: str) -> bool:
        """Check if a file exists and is readable."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        if not os.access(path, os.R_OK):
            logger.error(f"File not readable: {file_path}")
            return False
        return True

    def create_backup(self, file_path: str) -> str | None:
        """Create a backup of the file before modification."""
        if not self.backup:
            return None

        backup_path = f"{file_path}.backup"
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            return None

    def process_file(self, file_path: str) -> tuple[bool, int]:
        """
        Process a single file and replace import statements.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (success, number_of_replacements)
        """
        if not self.validate_file_exists(file_path):
            return False, 0

        try:
            # Read the file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            replacements_made = 0

            # Apply each import pattern replacement
            for pattern, replacement in self.import_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    replacements_made += len(matches)
                    if self.verbose:
                        logger.info(
                            f"  Replaced {len(matches)} instances of: {pattern}"
                        )

            # Only write if content changed
            if content != original_content:
                if self.dry_run:
                    logger.info(
                        f"[DRY RUN] Would modify {file_path} ({replacements_made} replacements)"
                    )
                    return True, replacements_made
                else:
                    # Create backup if requested
                    if self.backup:
                        self.create_backup(file_path)

                    # Write the modified content
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    logger.info(
                        f"Modified {file_path} ({replacements_made} replacements)"
                    )
                    return True, replacements_made
            else:
                if self.verbose:
                    logger.info(f"No changes needed in {file_path}")
                return True, 0

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False, 0

    def run(self) -> bool:
        """
        Run the import replacement process.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting import replacement process...")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"Backup: {'Enabled' if self.backup else 'Disabled'}")
        logger.info(f"Target files: {len(self.target_files)}")

        success = True

        for file_path in self.target_files:
            logger.info(f"Processing: {file_path}")

            file_success, replacements = self.process_file(file_path)

            self.stats["files_processed"] += 1
            if replacements > 0:
                self.stats["files_modified"] += 1
                self.stats["total_replacements"] += replacements

            if not file_success:
                self.stats["errors"] += 1
                success = False

        self._print_summary()
        return success

    def _print_summary(self) -> None:
        """Print a summary of the replacement process."""
        logger.info("\n" + "=" * 50)
        logger.info("REPLACEMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Total replacements: {self.stats['total_replacements']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.stats["errors"] > 0:
            logger.error("Some files had errors during processing")
        elif self.stats["files_modified"] > 0:
            logger.info("✅ Import replacement completed successfully")
        else:
            logger.info("No files required modifications")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Replace 'from src.' import statements with 'from crackseg.' in documentation files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files before modification",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    # Create the replacer
    replacer = ImportReplacer(
        dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose
    )

    # Run the replacement process
    success = replacer.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
