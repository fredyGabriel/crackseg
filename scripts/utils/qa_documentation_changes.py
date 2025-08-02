#!/usr/bin/env python3
"""
Documentation Changes QA Review

This script performs a comprehensive QA review of all documentation changes
made during the artifact system development process.

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


class DocumentationQAReviewer:
    """Performs comprehensive QA review of documentation changes."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            "files_reviewed": 0,
            "import_checks": 0,
            "reference_checks": 0,
            "syntax_checks": 0,
            "link_checks": 0,
            "backup_checks": 0,
            "issues_found": 0,
            "warnings": 0,
        }
        self.issues = []
        self.warnings = []

    def check_import_statements(self, content: str, file_path: str) -> bool:
        """Check that all import statements use correct format."""
        # Check for old src. imports in functional code
        old_imports = re.findall(r"from\s+src\.", content)
        if old_imports and "report" not in file_path.lower():
            self.issues.append(
                f"Old import format found in {file_path}: {len(old_imports)} instances"
            )
            self.stats["issues_found"] += 1
            return False

        # Check for correct crackseg. imports
        correct_imports = re.findall(r"from\s+crackseg\.", content)
        if correct_imports:
            self.stats["import_checks"] += 1
            if self.verbose:
                logger.info(f"  ✅ Import consistency in {file_path}")

        return True

    def check_reference_consistency(
        self, content: str, file_path: str
    ) -> bool:
        """Check that all references use consistent format."""
        # Check for old src. references in functional content
        old_refs = re.findall(r"_target_:\s*src\.", content)
        if old_refs:
            self.issues.append(
                f"Old target reference found in {file_path}: {len(old_refs)} instances"
            )
            self.stats["issues_found"] += 1
            return False

        # Check for old patch references
        old_patches = re.findall(r"@patch\('src\.", content)
        if old_patches:
            self.issues.append(
                f"Old patch reference found in {file_path}: {len(old_patches)} instances"
            )
            self.stats["issues_found"] += 1
            return False

        # Check for old command references
        old_commands = re.findall(r"python -m src\.", content)
        if old_commands:
            self.issues.append(
                f"Old command reference found in {file_path}: {len(old_commands)} instances"
            )
            self.stats["issues_found"] += 1
            return False

        self.stats["reference_checks"] += 1
        return True

    def check_syntax_correctness(self, content: str, file_path: str) -> bool:
        """Check code snippet syntax correctness."""
        # Extract code blocks
        code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)

        for i, block in enumerate(code_blocks):
            try:
                # Basic syntax check
                compile(block, "<string>", "exec")
            except SyntaxError as e:
                self.issues.append(
                    f"Syntax error in {file_path} block {i + 1}: {e}"
                )
                self.stats["issues_found"] += 1
                return False

        if code_blocks:
            self.stats["syntax_checks"] += 1
            if self.verbose:
                logger.info(f"  ✅ Syntax correctness in {file_path}")

        return True

    def check_link_validity(self, content: str, file_path: str) -> bool:
        """Check that all internal links are valid."""
        # Find all markdown links
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

        for link_text, link_url in links:
            # Check if internal links exist
            if link_url.startswith("./") or link_url.startswith("../"):
                target_path = os.path.join(
                    os.path.dirname(file_path), link_url
                )
                if not os.path.exists(target_path):
                    self.issues.append(
                        f"Broken link in {file_path}: {link_text} -> {link_url}"
                    )
                    self.stats["issues_found"] += 1
                    return False

        self.stats["link_checks"] += 1
        return True

    def check_backup_files(self, file_path: str) -> bool:
        """Check that backup files exist for modified files."""
        backup_path = f"{file_path}.backup"
        reference_backup = f"{file_path}.reference_backup"
        consistency_backup = f"{file_path}.consistency_backup"

        if (
            os.path.exists(backup_path)
            or os.path.exists(reference_backup)
            or os.path.exists(consistency_backup)
        ):
            self.stats["backup_checks"] += 1
            if self.verbose:
                logger.info(f"  ✅ Backup found for {file_path}")
            return True
        else:
            self.warnings.append(f"No backup found for {file_path}")
            self.stats["warnings"] += 1
            return False

    def review_file(self, file_path: str) -> bool:
        """Review a single file for QA compliance."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            if self.verbose:
                logger.info(f"Reviewing: {file_path}")

            all_passed = True

            # Run all QA checks
            if not self.check_import_statements(content, file_path):
                all_passed = False

            if not self.check_reference_consistency(content, file_path):
                all_passed = False

            if not self.check_syntax_correctness(content, file_path):
                all_passed = False

            if not self.check_link_validity(content, file_path):
                all_passed = False

            if not self.check_backup_files(file_path):
                all_passed = False

            self.stats["files_reviewed"] += 1
            return all_passed

        except Exception as e:
            logger.error(f"Error reviewing {file_path}: {e}")
            self.stats["issues_found"] += 1
            return False

    def review_directory(self, directory: str) -> None:
        """Review all markdown files in a directory."""
        logger.info(f"QA Reviewing directory: {directory}")

        # Find all markdown files
        md_files = []
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        logger.info(f"Found {len(md_files)} markdown files to review")

        # Review each file
        for file_path in md_files:
            self.review_file(file_path)

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print QA review summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOCUMENTATION QA REVIEW SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files reviewed: {self.stats['files_reviewed']}")
        logger.info(f"Import checks passed: {self.stats['import_checks']}")
        logger.info(
            f"Reference checks passed: {self.stats['reference_checks']}"
        )
        logger.info(f"Syntax checks passed: {self.stats['syntax_checks']}")
        logger.info(f"Link checks passed: {self.stats['link_checks']}")
        logger.info(f"Backup checks passed: {self.stats['backup_checks']}")
        logger.info(f"Issues found: {self.stats['issues_found']}")
        logger.info(f"Warnings: {self.stats['warnings']}")

        if self.stats["issues_found"] == 0:
            logger.info("✅ All QA checks passed")
        else:
            logger.warning(f"⚠️  {self.stats['issues_found']} issues found")

        if self.issues:
            logger.info("")
            logger.info("ISSUES FOUND:")
            for issue in self.issues:
                logger.error(f"  - {issue}")

        if self.warnings:
            logger.info("")
            logger.info("WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")


def main():
    """Main function for QA review."""
    parser = argparse.ArgumentParser(
        description="Perform QA review of documentation changes"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--directory",
        default="docs",
        help="Directory to review (default: docs)",
    )

    args = parser.parse_args()

    # Initialize reviewer
    reviewer = DocumentationQAReviewer(verbose=args.verbose)

    # Review directory
    reviewer.review_directory(args.directory)


if __name__ == "__main__":
    main()
