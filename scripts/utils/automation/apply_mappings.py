"""Script to apply mappings from the registry to project files.

This script automates the process of applying path mappings to files across
the project, ensuring consistency during refactors and migrations.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from crackseg.utils.mapping_registry import get_registry  # noqa: E402
from scripts.utils.common.io_utils import read_text, write_text  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_file_extensions(mapping_type: str) -> set[str]:
    """Get file extensions to process based on mapping type.

    Args:
        mapping_type: Type of mapping ('import', 'config', 'docs', 'artifact')

    Returns:
        Set of file extensions to process
    """
    if mapping_type == "import":
        return {".py"}
    elif mapping_type == "config":
        return {".yaml", ".yml"}
    elif mapping_type == "docs":
        return {".md", ".rst", ".txt"}
    elif mapping_type == "artifact":
        return {".py", ".yaml", ".yml", ".json", ".md"}
    else:
        return {".py", ".yaml", ".yml", ".md", ".txt", ".json"}


def should_skip_file(file_path: Path, skip_patterns: list[str]) -> bool:
    """Check if a file should be skipped based on patterns.

    Args:
        file_path: Path to the file to check
        skip_patterns: List of patterns to skip

    Returns:
        True if file should be skipped, False otherwise
    """
    file_str = str(file_path)
    return any(pattern in file_str for pattern in skip_patterns)


def apply_mappings_to_file(
    file_path: Path, registry, mapping_types: list[str], dry_run: bool = False
) -> bool:
    """Apply mappings to a single file.

    Args:
        file_path: Path to the file to process
        registry: Mapping registry instance
        mapping_types: List of mapping types to apply
        dry_run: If True, don't actually modify files

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read file content
        original_content = read_text(file_path)

        # Apply mappings
        modified_content = original_content
        for mapping_type in mapping_types:
            modified_content = registry.apply_mapping(
                modified_content, mapping_type
            )

        # Check if content changed
        if modified_content != original_content:
            if not dry_run:
                # Write modified content back to file
                write_text(file_path, modified_content)
                logging.info(f"Updated: {file_path}")
            else:
                logging.info(f"Would update: {file_path}")
            return True
        else:
            logging.debug(f"No changes needed: {file_path}")
            return False

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False


def process_directory(
    directory: Path,
    registry,
    mapping_types: list[str],
    file_extensions: set[str],
    skip_patterns: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process all files in a directory recursively.

    Args:
        directory: Directory to process
        registry: Mapping registry instance
        mapping_types: List of mapping types to apply
        file_extensions: Set of file extensions to process
        skip_patterns: List of patterns to skip
        dry_run: If True, don't actually modify files

    Returns:
        Tuple of (files_processed, files_modified)
    """
    files_processed = 0
    files_modified = 0

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        # Check file extension
        if file_path.suffix not in file_extensions:
            continue

        # Check skip patterns
        if should_skip_file(file_path, skip_patterns):
            logging.debug(f"Skipping: {file_path}")
            continue

        files_processed += 1
        if apply_mappings_to_file(file_path, registry, mapping_types, dry_run):
            files_modified += 1

    return files_processed, files_modified


def main() -> None:
    """Main function to run the mapping application script."""
    parser = argparse.ArgumentParser(
        description="Apply mappings from registry to project files"
    )
    parser.add_argument(
        "--mapping-types",
        nargs="+",
        choices=["import", "config", "docs", "artifact", "checkpoint"],
        default=["import", "config", "docs", "artifact"],
        help="Mapping types to apply",
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["src", "configs", "docs", "scripts"],
        help="Directories to process",
    )
    parser.add_argument(
        "--skip-patterns",
        nargs="+",
        default=[
            "__pycache__",
            ".git",
            "artifacts",
            "venv",
            ".venv",
            "node_modules",
            ".pytest_cache",
        ],
        help="Patterns to skip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually changing files",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Get registry
    registry = get_registry()

    # Validate mappings
    errors = registry.validate_mappings()
    if errors:
        logger.error("Registry validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Get statistics
    stats = registry.get_statistics()
    logger.info(f"Registry statistics: {stats}")

    # Determine file extensions to process
    file_extensions = set()
    for mapping_type in args.mapping_types:
        file_extensions.update(get_file_extensions(mapping_type))

    logger.info(f"Processing files with extensions: {file_extensions}")
    logger.info(f"Mapping types: {args.mapping_types}")

    total_processed = 0
    total_modified = 0

    # Process each directory
    for directory_str in args.directories:
        directory = Path(directory_str)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        logger.info(f"Processing directory: {directory}")
        processed, modified = process_directory(
            directory,
            registry,
            args.mapping_types,
            file_extensions,
            args.skip_patterns,
            args.dry_run,
        )

        total_processed += processed
        total_modified += modified

        logger.info(f"  Processed: {processed}, Modified: {modified}")

    # Summary
    logger.info(
        f"Summary: {total_processed} files processed, {total_modified} files modified"
    )

    if args.dry_run:
        logger.info("Dry run completed - no files were actually modified")
    else:
        logger.info("Mapping application completed")


if __name__ == "__main__":
    main()
