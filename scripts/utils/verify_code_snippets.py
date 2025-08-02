#!/usr/bin/env python3
"""
Code Snippet Syntax Verifier

This script verifies the syntax correctness of code snippets in documentation
files, particularly focusing on import statements and code examples.

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import argparse
import ast
import logging
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeSnippetVerifier:
    """Verifies syntax correctness of code snippets in documentation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "snippets_found": 0,
            "syntax_errors": 0,
            "import_errors": 0,
            "warnings": 0,
        }

    def extract_code_snippets(
        self, content: str
    ) -> list[tuple[str, int, str]]:
        """Extract code snippets from markdown content."""
        snippets = []

        # Pattern to match code blocks
        code_block_pattern = r"```(\w+)?\n(.*?)```"

        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            language = match.group(1) or ""
            code = match.group(2).strip()
            line_number = content[: match.start()].count("\n") + 1

            if language.lower() in ["python", "py", ""]:
                snippets.append((code, line_number, language))

        return snippets

    def verify_python_syntax(
        self, code: str, file_path: str, line_number: int
    ) -> bool:
        """Verify Python syntax of a code snippet."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}:{line_number}: {e}")
            self.stats["syntax_errors"] += 1
            return False
        except Exception as e:
            logger.warning(
                f"Unexpected error parsing {file_path}:{line_number}: {e}"
            )
            self.stats["warnings"] += 1
            return False

    def check_import_statements(
        self, code: str, file_path: str, line_number: int
    ) -> bool:
        """Check import statements for correctness."""
        import_pattern = r"from\s+crackseg\.[\w\.]+\s+import\s+[\w\s,]+"
        imports = re.findall(import_pattern, code)

        for import_stmt in imports:
            try:
                # Try to parse the import statement
                ast.parse(import_stmt)
                if self.verbose:
                    logger.info(f"  ✅ Valid import: {import_stmt}")
            except SyntaxError as e:
                logger.error(
                    f"Invalid import in {file_path}:{line_number}: {import_stmt} - {e}"
                )
                self.stats["import_errors"] += 1
                return False

        return True

    def process_file(self, file_path: str) -> bool:
        """Process a single file for code snippet verification."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Extract code snippets
            snippets = self.extract_code_snippets(content)

            if self.verbose:
                logger.info(
                    f"Found {len(snippets)} code snippets in {file_path}"
                )

            all_valid = True

            for code, line_number, _language in snippets:
                self.stats["snippets_found"] += 1

                # Verify Python syntax
                if not self.verify_python_syntax(code, file_path, line_number):
                    all_valid = False

                # Check import statements
                if not self.check_import_statements(
                    code, file_path, line_number
                ):
                    all_valid = False

            self.stats["files_processed"] += 1
            return all_valid

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
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
        """Print verification summary."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("CODE SNIPPET VERIFICATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Code snippets found: {self.stats['snippets_found']}")
        logger.info(f"Syntax errors: {self.stats['syntax_errors']}")
        logger.info(f"Import errors: {self.stats['import_errors']}")
        logger.info(f"Warnings: {self.stats['warnings']}")

        if (
            self.stats["syntax_errors"] == 0
            and self.stats["import_errors"] == 0
        ):
            logger.info("✅ All code snippets are syntactically correct")
        else:
            logger.warning("⚠️  Some code snippets have syntax errors")


def main():
    """Main function for code snippet verification."""
    parser = argparse.ArgumentParser(
        description="Verify syntax correctness of code snippets in documentation"
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
    verifier = CodeSnippetVerifier(verbose=args.verbose)

    # Process directory
    verifier.process_directory(args.directory)


if __name__ == "__main__":
    main()
