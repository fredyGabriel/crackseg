"""CI script to detect stale reports and documentation.

This script checks for stale documentation, tickets, and analysis reports
that may have outdated paths or references based on the mapping registry.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))

from crackseg.utils.mapping_registry import get_registry  # noqa: E402
from scripts.utils.common.io_utils import read_text  # noqa: E402


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def check_file_for_stale_references(
    file_path: Path, registry, mapping_types: list[str]
) -> list[dict[str, Any]]:
    """Check a file for stale references based on mapping registry.

    Args:
        file_path: Path to the file to check
        registry: Mapping registry instance
        mapping_types: List of mapping types to check

    Returns:
        List of issues found in the file
    """
    issues = []

    try:
        content = read_text(file_path)

        for mapping_type in mapping_types:
            mappings = registry.get_mappings_by_type(mapping_type)

            for mapping in mappings:
                if mapping.deprecated:
                    continue

                # Check if old path is still referenced
                if mapping.old_path in content:
                    issues.append(
                        {
                            "file": str(file_path),
                            "mapping_type": mapping_type,
                            "old_path": mapping.old_path,
                            "new_path": mapping.new_path,
                            "description": mapping.description,
                            "severity": "warning",
                        }
                    )

        return issues

    except Exception as e:
        logging.error(f"Error checking {file_path}: {e}")
        return [{"file": str(file_path), "error": str(e), "severity": "error"}]


def check_documentation_files(registry) -> list[dict[str, Any]]:
    """Check documentation files for stale references.

    Args:
        registry: Mapping registry instance

    Returns:
        List of issues found in documentation
    """
    issues = []
    docs_dir = Path("docs")

    if not docs_dir.exists():
        logging.warning("docs/ directory not found")
        return issues

    # Check markdown files
    for file_path in docs_dir.rglob("*.md"):
        file_issues = check_file_for_stale_references(
            file_path, registry, ["docs", "import", "config", "artifact"]
        )
        issues.extend(file_issues)

    return issues


def check_config_files(registry) -> list[dict[str, Any]]:
    """Check configuration files for stale references.

    Args:
        registry: Mapping registry instance

    Returns:
        List of issues found in configuration files
    """
    issues = []
    configs_dir = Path("configs")

    if not configs_dir.exists():
        logging.warning("configs/ directory not found")
        return issues

    # Check YAML files
    for file_path in configs_dir.rglob("*.yaml"):
        file_issues = check_file_for_stale_references(
            file_path, registry, ["config", "artifact"]
        )
        issues.extend(file_issues)

    for file_path in configs_dir.rglob("*.yml"):
        file_issues = check_file_for_stale_references(
            file_path, registry, ["config", "artifact"]
        )
        issues.extend(file_issues)

    return issues


def check_source_files(registry) -> list[dict[str, Any]]:
    """Check source files for stale references.

    Args:
        registry: Mapping registry instance

    Returns:
        List of issues found in source files
    """
    issues = []
    src_dir = Path("src")

    if not src_dir.exists():
        logging.warning("src/ directory not found")
        return issues

    # Check Python files
    for file_path in src_dir.rglob("*.py"):
        file_issues = check_file_for_stale_references(
            file_path, registry, ["import", "config", "artifact"]
        )
        issues.extend(file_issues)

    return issues


def check_script_files(registry) -> list[dict[str, Any]]:
    """Check script files for stale references.

    Args:
        registry: Mapping registry instance

    Returns:
        List of issues found in script files
    """
    issues = []
    scripts_dir = Path("scripts")

    if not scripts_dir.exists():
        logging.warning("scripts/ directory not found")
        return issues

    # Check Python files in scripts
    for file_path in scripts_dir.rglob("*.py"):
        file_issues = check_file_for_stale_references(
            file_path, registry, ["import", "config", "artifact"]
        )
        issues.extend(file_issues)

    return issues


def generate_report(issues: list[dict[str, Any]]) -> str:
    """Generate a human-readable report from issues.

    Args:
        issues: List of issues found

    Returns:
        Formatted report string
    """
    if not issues:
        return "✅ No stale references found!"

    report_lines = ["🔍 Stale References Report", "=" * 50, ""]

    # Group by severity
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if errors:
        report_lines.append("❌ ERRORS:")
        for issue in errors:
            if "error" in issue:
                report_lines.append(f"  - {issue['file']}: {issue['error']}")
            else:
                report_lines.append(
                    f"  - {issue['file']}: {issue['old_path']} -> {issue['new_path']} "
                    f"({issue['mapping_type']})"
                )
        report_lines.append("")

    if warnings:
        report_lines.append("⚠️  WARNINGS:")
        for issue in warnings:
            report_lines.append(
                f"  - {issue['file']}: {issue['old_path']} -> {issue['new_path']} "
                f"({issue['mapping_type']}) - {issue['description']}"
            )
        report_lines.append("")

    # Summary
    report_lines.append(
        f"Summary: {len(errors)} errors, {len(warnings)} warnings"
    )

    return "\n".join(report_lines)


def main() -> int:
    """Main function to run the stale reports check.

    Returns:
        Exit code (0 for success, 1 for issues found)
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get registry
    registry = get_registry()

    # Validate mappings
    errors = registry.validate_mappings()
    if errors:
        logger.error("Registry validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return 1

    # Get statistics
    stats = registry.get_statistics()
    logger.info(f"Registry statistics: {stats}")

    # Check different file types
    all_issues = []

    logger.info("Checking documentation files...")
    all_issues.extend(check_documentation_files(registry))

    logger.info("Checking configuration files...")
    all_issues.extend(check_config_files(registry))

    logger.info("Checking source files...")
    all_issues.extend(check_source_files(registry))

    logger.info("Checking script files...")
    all_issues.extend(check_script_files(registry))

    # Generate and print report
    report = generate_report(all_issues)
    print(report)

    # Return appropriate exit code
    errors = [i for i in all_issues if i.get("severity") == "error"]
    warnings = [i for i in all_issues if i.get("severity") == "warning"]

    if errors:
        logger.error(f"Found {len(errors)} errors")
        return 1
    elif warnings:
        logger.warning(f"Found {len(warnings)} warnings")
        return 0  # Warnings don't fail CI
    else:
        logger.info("No issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
