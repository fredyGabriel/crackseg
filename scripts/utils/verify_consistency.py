#!/usr/bin/env python3
"""
Documentation Consistency Verifier

This script verifies consistency across all documentation files, checking:
1. Import statement consistency (from crackseg.)
2. Code snippet syntax consistency
3. File naming consistency
4. Directory structure consistency
5. Cross-reference consistency

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import argparse
import logging
import os
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsistencyVerifier:
    """Verifies consistency across documentation files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "import_consistency": 0,
            "syntax_consistency": 0,
            "naming_consistency": 0,
            "structure_consistency": 0,
            "cross_reference_consistency": 0,
            "errors": 0,
            "warnings": 0,
        }
        self.issues = []

    def check_import_consistency(self, content: str, file_path: str) -> bool:
        """Check that all imports use consistent format."""
        # Check for old src. imports
        old_imports = re.findall(r"from\s+src\.", content)
        if old_imports:
            self.issues.append(
                f"Old import format found in {file_path}: {len(old_imports)} instances"
            )
            self.stats["errors"] += 1
            return False

        # Check for correct crackseg. imports
        correct_imports = re.findall(r"from\s+crackseg\.", content)
        if correct_imports:
            self.stats["import_consistency"] += 1
            if self.verbose:
                logger.info(f"  ✅ Import consistency in {file_path}")

        return True

    def check_syntax_consistency(self, content: str, file_path: str) -> bool:
        """Check code snippet syntax consistency."""
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
                self.stats["errors"] += 1
                return False

        if code_blocks:
            self.stats["syntax_consistency"] += 1
            if self.verbose:
                logger.info(f"  ✅ Syntax consistency in {file_path}")

        return True

    def check_naming_consistency(self, file_path: str) -> bool:
        """Check file naming consistency."""
        filename = os.path.basename(file_path)

        # Check for consistent naming patterns
        if not re.match(r"^[a-z0-9_]+\.md$", filename):
            self.issues.append(f"Inconsistent filename format: {filename}")
            self.stats["warnings"] += 1
            return False

        self.stats["naming_consistency"] += 1
        return True

    def check_structure_consistency(self, file_path: str) -> bool:
        """Check directory structure consistency."""
        path_parts = Path(file_path).parts

        # Check for expected directory structure
        if len(path_parts) < 3:  # Should be at least docs/category/file.md
            self.issues.append(f"Unexpected file location: {file_path}")
            self.stats["warnings"] += 1
            return False

        self.stats["structure_consistency"] += 1
        return True

    def check_cross_references(self, content: str, file_path: str) -> bool:
        """Check cross-reference consistency."""
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
                    self.stats["errors"] += 1
                    return False

        self.stats["cross_reference_consistency"] += 1
        return True

    def process_file(self, file_path: str) -> bool:
        """Process a single file for consistency verification."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            if self.verbose:
                logger.info(f"Processing: {file_path}")

            all_consistent = True

            # Run all consistency checks
            if not self.check_import_consistency(content, file_path):
                all_consistent = False

            if not self.check_syntax_consistency(content, file_path):
                all_consistent = False

            if not self.check_naming_consistency(file_path):
                all_consistent = False

            if not self.check_structure_consistency(file_path):
                all_consistent = False

            if not self.check_cross_references(content, file_path):
                all_consistent = False

            self.stats["files_processed"] += 1
            return all_consistent

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
            self.process_file(file_path)

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print consistency verification summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOCUMENTATION CONSISTENCY VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Import consistency: {self.stats['import_consistency']}")
        logger.info(f"Syntax consistency: {self.stats['syntax_consistency']}")
        logger.info(f"Naming consistency: {self.stats['naming_consistency']}")
        logger.info(
            f"Structure consistency: {self.stats['structure_consistency']}"
        )
        logger.info(
            f"Cross-reference consistency: {self.stats['cross_reference_consistency']}"
        )
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Warnings: {self.stats['warnings']}")

        if self.stats["errors"] == 0:
            logger.info("✅ All consistency checks passed")
        else:
            logger.warning(
                f"⚠️  {self.stats['errors']} consistency issues found"
            )

        if self.issues:
            logger.info("")
            logger.info("ISSUES FOUND:")
            for issue in self.issues:
                logger.warning(f"  - {issue}")


def main():
    """Main function for consistency verification."""
    parser = argparse.ArgumentParser(
        description="Verify consistency across documentation files"
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

    # Initialize verifier
    verifier = ConsistencyVerifier(verbose=args.verbose)

    # Process directory
    verifier.process_directory(args.directory)


if __name__ == "__main__":
    main()
