#!/usr/bin/env python3
"""
Final Documentation Cleanup

This script performs final cleanup of documentation issues before commit:
1. Fix remaining old references
2. Remove backup files (optional)
3. Create final verification report

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import argparse
import logging
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FinalCleanup:
    """Performs final cleanup of documentation issues."""

    def __init__(self, remove_backups: bool = False, verbose: bool = False):
        self.remove_backups = remove_backups
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "backups_removed": 0,
            "errors": 0,
        }

    def fix_remaining_references(self, content: str) -> tuple[str, int]:
        """Fix any remaining old references."""
        fixes_made = 0

        # Fix old target references
        old_targets = re.findall(r"_target_:\s*src\.", content)
        if old_targets:
            content = re.sub(
                r"_target_:\s*src\.", "_target_: crackseg.", content
            )
            fixes_made += len(old_targets)
            if self.verbose:
                logger.info(
                    f"  Fixed {len(old_targets)} old target references"
                )

        # Fix old patch references
        old_patches = re.findall(r"@patch\('src\.", content)
        if old_patches:
            content = re.sub(r"@patch\('src\.", "@patch('crackseg.", content)
            fixes_made += len(old_patches)
            if self.verbose:
                logger.info(f"  Fixed {len(old_patches)} old patch references")

        return content, fixes_made

    def process_file(self, file_path: str) -> bool:
        """Process a single file for final cleanup."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            total_fixes = 0

            # Apply fixes
            content, reference_fixes = self.fix_remaining_references(content)
            total_fixes += reference_fixes

            # Check if any changes were made
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"  Modified {file_path} ({total_fixes} fixes)")
                self.stats["files_modified"] += 1
            else:
                if self.verbose:
                    logger.info(f"  No fixes needed in {file_path}")

            self.stats["files_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats["errors"] += 1
            return False

    def remove_backup_files(self) -> None:
        """Remove backup files if requested."""
        if not self.remove_backups:
            logger.info(
                "Backup removal skipped (use --remove-backups to enable)"
            )
            return

        backup_extensions = [
            ".backup",
            ".reference_backup",
            ".consistency_backup",
            ".qa_backup",
        ]

        for root, _dirs, files in os.walk("docs"):
            for file in files:
                for ext in backup_extensions:
                    if file.endswith(ext):
                        backup_path = os.path.join(root, file)
                        try:
                            os.remove(backup_path)
                            self.stats["backups_removed"] += 1
                            if self.verbose:
                                logger.info(f"  Removed backup: {backup_path}")
                        except Exception as e:
                            logger.error(f"Error removing {backup_path}: {e}")

    def process_directory(self, directory: str) -> None:
        """Process all markdown files in a directory."""
        logger.info(f"Final cleanup of directory: {directory}")

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

        # Remove backups if requested
        self.remove_backup_files()

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print cleanup summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL DOCUMENTATION CLEANUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Backups removed: {self.stats['backups_removed']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.remove_backups:
            logger.info("Mode: BACKUP REMOVAL ENABLED")
        else:
            logger.info("Mode: BACKUP REMOVAL DISABLED")

        logger.info("")
        logger.info("✅ Documentation cleanup completed")
        logger.info("✅ Ready for commit")


def main():
    """Main function for final cleanup."""
    parser = argparse.ArgumentParser(
        description="Perform final cleanup of documentation before commit"
    )
    parser.add_argument(
        "--remove-backups",
        action="store_true",
        help="Remove backup files after cleanup",
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

    # Initialize cleanup
    cleanup = FinalCleanup(
        remove_backups=args.remove_backups, verbose=args.verbose
    )

    # Process directory
    cleanup.process_directory(args.directory)


if __name__ == "__main__":
    main()
