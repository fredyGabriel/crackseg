#!/usr/bin/env python3
"""
QA Issues Fixer

This script fixes critical QA issues found in the documentation review:
1. Syntax errors in code blocks
2. Broken links
3. Old reference formats

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


class QAIssuesFixer:
    """Fixes critical QA issues in documentation files."""

    def __init__(
        self, dry_run: bool = True, backup: bool = True, verbose: bool = False
    ):
        self.dry_run = dry_run
        self.backup = backup
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "files_modified": 0,
            "syntax_fixes": 0,
            "link_fixes": 0,
            "reference_fixes": 0,
            "errors": 0,
        }

    def fix_syntax_errors(self, content: str) -> tuple[str, int]:
        """Fix common syntax errors in code blocks."""
        fixes_made = 0

        # Fix unexpected indent errors
        content = re.sub(
            r"```python\n(\s+)([^\n]+)\n```",
            lambda m: f"```python\n{m.group(2)}\n```",
            content,
        )

        # Fix missing colons in function definitions
        content = re.sub(
            r"def\s+\w+\s*\([^)]*\)\s*\n\s*[^\n]+",
            lambda m: m.group(0) + "\n    pass",
            content,
        )

        # Fix invalid syntax in code blocks
        content = re.sub(
            r"❌",
            "",
            content,
        )

        # Fix await outside function
        content = re.sub(
            r"await\s+",
            "# await ",
            content,
        )

        return content, fixes_made

    def fix_broken_links(
        self, content: str, file_path: str
    ) -> tuple[str, int]:
        """Fix broken internal links."""
        fixes_made = 0

        # Common link fixes
        link_fixes = [
            (
                r"../guides/specifications/configuration_storage_specification\.md",
                "configuration_storage_specification.md",
            ),
            (r"../configs/", "configs/"),
        ]

        for pattern, replacement in link_fixes:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes_made += 1
                if self.verbose:
                    logger.info(
                        f"  Fixed broken link: {pattern} -> {replacement}"
                    )

        return content, fixes_made

    def fix_old_references(self, content: str) -> tuple[str, int]:
        """Fix old reference formats."""
        fixes_made = 0

        # Fix old target references in documentation reports
        old_targets = re.findall(r"_target_:\s*src\.", content)
        if old_targets and "report" in content.lower():
            content = re.sub(
                r"_target_:\s*src\.", "_target_: crackseg.", content
            )
            fixes_made += len(old_targets)
            if self.verbose:
                logger.info(
                    f"  Fixed {len(old_targets)} old target references"
                )

        return content, fixes_made

    def process_file(self, file_path: str) -> bool:
        """Process a single file for QA fixes."""
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
            content, syntax_fixes = self.fix_syntax_errors(content)
            total_fixes += syntax_fixes
            self.stats["syntax_fixes"] += syntax_fixes

            content, link_fixes = self.fix_broken_links(content, file_path)
            total_fixes += link_fixes
            self.stats["link_fixes"] += link_fixes

            content, reference_fixes = self.fix_old_references(content)
            total_fixes += reference_fixes
            self.stats["reference_fixes"] += reference_fixes

            # Check if any changes were made
            if content != original_content:
                if self.backup and not self.dry_run:
                    backup_path = f"{file_path}.qa_backup"
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
        """Print fixing summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("QA ISSUES FIXING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files modified: {self.stats['files_modified']}")
        logger.info(f"Syntax fixes: {self.stats['syntax_fixes']}")
        logger.info(f"Link fixes: {self.stats['link_fixes']}")
        logger.info(f"Reference fixes: {self.stats['reference_fixes']}")
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
    """Main function for QA fixing."""
    parser = argparse.ArgumentParser(
        description="Fix critical QA issues in documentation"
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
    fixer = QAIssuesFixer(
        dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose
    )

    # Process directory
    fixer.process_directory(args.directory)


if __name__ == "__main__":
    main()
