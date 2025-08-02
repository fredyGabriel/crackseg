#!/usr/bin/env python3
"""
Documentation References Updater

This script updates documentation indexes and references that still contain
old 'src.' references to use the new 'crackseg.' format.

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


class ReferenceUpdater:
    """Updates documentation references from src. to crackseg."""

    def __init__(
        self, dry_run: bool = True, backup: bool = True, verbose: bool = False
    ):
        self.dry_run = dry_run
        self.backup = backup
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "target_fixes": 0,
            "patch_fixes": 0,
            "command_fixes": 0,
            "errors": 0,
        }

    def fix_target_references(self, content: str) -> tuple[str, int]:
        """Fix _target_ references from src. to crackseg."""
        fixes_made = 0

        # Fix _target_ references
        target_pattern = r"_target_:\s*src\.(crackseg\.)?"
        if re.search(target_pattern, content):
            content = re.sub(target_pattern, "_target_: crackseg.", content)
            fixes_made += 1
            if self.verbose:
                logger.info("  Fixed _target_ references")

        return content, fixes_made

    def fix_patch_references(self, content: str) -> tuple[str, int]:
        """Fix @patch references from src. to crackseg."""
        fixes_made = 0

        # Fix @patch references
        patch_pattern = r"@patch\('src\.(crackseg\.)?"
        if re.search(patch_pattern, content):
            content = re.sub(patch_pattern, "@patch('crackseg.", content)
            fixes_made += 1
            if self.verbose:
                logger.info("  Fixed @patch references")

        return content, fixes_made

    def fix_command_references(self, content: str) -> tuple[str, int]:
        """Fix command references from src. to crackseg."""
        fixes_made = 0

        # Fix python -m src. commands
        command_pattern = r"python -m src\.(crackseg\.)?"
        if re.search(command_pattern, content):
            content = re.sub(command_pattern, "python -m crackseg.", content)
            fixes_made += 1
            if self.verbose:
                logger.info("  Fixed command references")

        return content, fixes_made

    def fix_module_references(self, content: str) -> tuple[str, int]:
        """Fix module references from src. to crackseg."""
        fixes_made = 0

        # Fix module references in code blocks
        module_pattern = r"`src\.(crackseg\.)?"
        if re.search(module_pattern, content):
            content = re.sub(module_pattern, "`crackseg.", content)
            fixes_made += 1
            if self.verbose:
                logger.info("  Fixed module references")

        return content, fixes_made

    def process_file(self, file_path: str) -> bool:
        """Process a single file for reference updates."""
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
            content, target_fixes = self.fix_target_references(content)
            total_fixes += target_fixes
            self.stats["target_fixes"] += target_fixes

            content, patch_fixes = self.fix_patch_references(content)
            total_fixes += patch_fixes
            self.stats["patch_fixes"] += patch_fixes

            content, command_fixes = self.fix_command_references(content)
            total_fixes += command_fixes
            self.stats["command_fixes"] += command_fixes

            content, module_fixes = self.fix_module_references(content)
            total_fixes += module_fixes

            # Check if any changes were made
            if content != original_content:
                if self.backup and not self.dry_run:
                    backup_path = f"{file_path}.reference_backup"
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"  Created backup: {backup_path}")

                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(
                        f"  Modified {file_path} ({total_fixes} fixes)"
                    )
                else:
                    logger.info(
                        f"  Would modify {file_path} ({total_fixes} fixes)"
                    )

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
        """Print updating summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOCUMENTATION REFERENCES UPDATING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Target fixes: {self.stats['target_fixes']}")
        logger.info(f"Patch fixes: {self.stats['patch_fixes']}")
        logger.info(f"Command fixes: {self.stats['command_fixes']}")
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
    """Main function for reference updating."""
    parser = argparse.ArgumentParser(
        description="Update documentation references from src. to crackseg."
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

    # Initialize updater
    updater = ReferenceUpdater(
        dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose
    )

    # Process directory
    updater.process_directory(args.directory)


if __name__ == "__main__":
    main()
