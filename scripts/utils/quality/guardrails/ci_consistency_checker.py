"""CI Consistency Checker for Cross-Plan Automation.

This script integrates all consistency checks for CI/CD pipelines:
- Stale reports and documentation detection
- Broken link validation
- Import policy enforcement
- Mapping registry validation

This is the main entry point for CI systems to ensure project consistency.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
# Add project root so "scripts" is importable
sys.path.insert(0, str(project_root))  # noqa: E402
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))
sys.path.insert(
    0, str(project_root / "scripts" / "utils" / "quality" / "guardrails")
)

from import_policy_checker import check_directory_imports  # noqa: E402
from link_checker import check_directory_links  # noqa: E402
from simple_mapping_registry import (  # noqa: E402
    SimpleMappingRegistry,
    create_default_registry,
)

from scripts.utils.common.io_utils import read_text  # noqa: E402
from scripts.utils.common.logging_utils import setup_logging  # noqa: E402


def check_stale_reports(
    registry: SimpleMappingRegistry, directories: list[Path]
) -> dict[str, Any]:
    """Check for stale reports and documentation.

    Args:
        registry: Mapping registry for validation
        directories: Directories to check

    Returns:
        Dictionary with check results
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Checking for stale reports and documentation...")

    stale_issues = []

    for directory in directories:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        # Check for files that reference old paths
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in {
                ".md",
                ".py",
                ".yaml",
                ".yml",
                ".txt",
            }:
                try:
                    content = read_text(file_path)

                    # Check for old path references
                    for mapping in registry.mappings:
                        if (
                            mapping.old_path in content
                            and not mapping.deprecated
                        ):
                            stale_issues.append(
                                {
                                    "file": str(file_path),
                                    "old_path": mapping.old_path,
                                    "new_path": mapping.new_path,
                                    "mapping_type": mapping.mapping_type,
                                    "description": mapping.description,
                                }
                            )

                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

    return {
        "stale_count": len(stale_issues),
        "stale_issues": stale_issues,
    }


def check_mapping_registry_consistency(
    registry: SimpleMappingRegistry,
) -> dict[str, Any]:
    """Check mapping registry for consistency issues.

    Args:
        registry: Mapping registry to validate

    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validating mapping registry consistency...")

    # Validate mappings
    validation_errors = registry.validate_mappings()

    # Get statistics
    stats = registry.get_statistics()

    return {
        "validation_errors": validation_errors,
        "validation_error_count": len(validation_errors),
        "statistics": stats,
    }


def generate_consistency_report(
    link_results: dict[str, Any],
    import_results: dict[str, Any],
    stale_results: dict[str, Any],
    registry_results: dict[str, Any],
    duplicate_results: dict[str, Any],
) -> str:
    """Generate a comprehensive consistency report.

    Args:
        link_results: Results from link checking
        import_results: Results from import policy checking
        stale_results: Results from stale report checking
        registry_results: Results from registry validation

    Returns:
        Formatted report string
    """
    report_lines = [
        "🔗 CI Consistency Checker Report",
        "=" * 60,
        "",
    ]

    # Link checker results
    if "issues" in link_results:
        report_lines.append("📎 LINK CHECKER:")
        report_lines.append(f"  - Issues found: {len(link_results['issues'])}")
        errors = [
            i for i in link_results["issues"] if i.get("severity") == "error"
        ]
        warnings = [
            i for i in link_results["issues"] if i.get("severity") == "warning"
        ]
        report_lines.append(
            f"  - Errors: {len(errors)}, Warnings: {len(warnings)}"
        )
        report_lines.append("")

    # Import policy results
    if "violations" in import_results:
        report_lines.append("📦 IMPORT POLICY:")
        report_lines.append(
            f"  - Violations found: {len(import_results['violations'])}"
        )
        report_lines.append("")

    # Stale reports results
    report_lines.append("📄 STALE REPORTS:")
    report_lines.append(
        f"  - Stale references found: {stale_results['stale_count']}"
    )
    report_lines.append("")

    # Duplicate code results
    report_lines.append("🧬 DUPLICATE CODE:")
    report_lines.append(
        f"  - Current groups: {duplicate_results.get('current', 0)}"
    )
    report_lines.append(
        f"  - Baseline groups: {duplicate_results.get('baseline', 0)}"
    )
    report_lines.append(
        f"  - New vs. baseline: {duplicate_results.get('new', 0)}"
    )
    report_lines.append("")

    # Registry validation results
    report_lines.append("🗂️  MAPPING REGISTRY:")
    report_lines.append(
        f"  - Validation errors: {registry_results['validation_error_count']}"
    )
    stats = registry_results.get("statistics", {})
    report_lines.append(
        f"  - Total mappings: {stats.get('total_mappings', 0)}"
    )
    report_lines.append(
        f"  - Deprecated mappings: {stats.get('deprecated_count', 0)}"
    )
    report_lines.append("")

    # Summary
    total_errors = (
        len(
            [
                i
                for i in link_results.get("issues", [])
                if i.get("severity") == "error"
            ]
        )
        + len(import_results.get("violations", []))
        + registry_results["validation_error_count"]
        + (1 if duplicate_results.get("new", 0) > 0 else 0)
    )

    total_warnings = (
        len(
            [
                i
                for i in link_results.get("issues", [])
                if i.get("severity") == "warning"
            ]
        )
        + stale_results["stale_count"]
    )

    report_lines.append("📊 SUMMARY:")
    report_lines.append(f"  - Total Errors: {total_errors}")
    report_lines.append(f"  - Total Warnings: {total_warnings}")

    if total_errors == 0:
        report_lines.append("  - ✅ All critical checks passed!")
    else:
        report_lines.append("  - ❌ Critical issues found!")

    return "\n".join(report_lines)


def _compute_duplicate_results() -> dict[str, Any]:
    """Run duplicate guardrail in check mode and summarize counts."""
    project_root = Path(__file__).resolve().parents[4]
    guardrail = (
        project_root
        / "scripts"
        / "utils"
        / "quality"
        / "guardrails"
        / "duplicate_guardrail.py"
    )
    reports_dir = project_root / "docs" / "reports" / "project-reports"
    current_json = reports_dir / "duplicate_scan_report.json"
    baseline_json = reports_dir / "duplicate_scan_baseline.json"

    # Run guardrail (no baseline update) allowing 0 new groups
    try:
        subprocess.run(
            [sys.executable, str(guardrail), "--max-delta", "0"],
            check=False,
        )
    except Exception:
        pass

    from scripts.utils.common.io_utils import read_json  # noqa: E402

    def _hashes(p: Path) -> set[str]:
        try:
            data = read_json(p)
            return {
                str(item.get("hash", "")) for item in data if "hash" in item
            }
        except Exception:
            return set()

    current = _hashes(current_json)
    baseline = _hashes(baseline_json)
    return {
        "current": len(current),
        "baseline": len(baseline),
        "new": len(current - baseline),
    }


def main() -> int:
    """Main function to run all consistency checks.

    Returns:
        Exit code (0 for success, 1 for critical issues found)
    """
    parser = argparse.ArgumentParser(
        description="Run comprehensive consistency checks for CI/CD"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["docs", "scripts", "src"],
        help="Directories to check for consistency issues",
    )
    parser.add_argument(
        "--skip-links",
        action="store_true",
        help="Skip link checking",
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip import policy checking",
    )
    parser.add_argument(
        "--skip-stale",
        action="store_true",
        help="Skip stale report checking",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip duplicate code guardrail",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail CI on warnings (not just errors)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Get registry
    registry = create_default_registry()

    # Convert directory strings to Path objects
    directories = [Path(d) for d in args.directories]

    # Initialize results
    link_results = {"issues": []}
    import_results = {"violations": []}
    stale_results = {"stale_count": 0, "stale_issues": []}
    registry_results = {"validation_error_count": 0, "statistics": {}}
    duplicate_results = {"current": 0, "baseline": 0, "new": 0}

    # Run link checker
    if not args.skip_links:
        logger.info("🔗 Running link checker...")
        all_link_issues = []
        for directory in directories:
            if directory.exists():
                directory_issues = check_directory_links(directory, registry)
                all_link_issues.extend(directory_issues)
        link_results["issues"] = all_link_issues

    # Run import policy checker
    if not args.skip_imports:
        logger.info("📦 Running import policy checker...")
        all_import_violations = []
        for directory in directories:
            if directory.exists():
                directory_violations = check_directory_imports(
                    directory, registry
                )
                all_import_violations.extend(directory_violations)
        import_results["violations"] = all_import_violations

    # Run stale report checker
    if not args.skip_stale:
        logger.info("📄 Running stale report checker...")
        stale_results = check_stale_reports(registry, directories)

    # Run registry validation
    logger.info("🗂️  Running registry validation...")
    registry_results = check_mapping_registry_consistency(registry)

    # Run duplicate guardrail
    if not args.skip_duplicates:
        logger.info("🧬 Running duplicate code guardrail...")
        duplicate_results = _compute_duplicate_results()

    # Generate and print report
    report = generate_consistency_report(
        link_results,
        import_results,
        stale_results,
        registry_results,
        duplicate_results,
    )
    print(report)

    # Determine exit code
    errors = [
        i for i in link_results["issues"] if i.get("severity") == "error"
    ]
    warnings = [
        i for i in link_results["issues"] if i.get("severity") == "warning"
    ]

    total_errors = (
        len(errors)
        + len(import_results["violations"])
        + registry_results["validation_error_count"]
        + (1 if duplicate_results.get("new", 0) > 0 else 0)
    )

    total_warnings = len(warnings) + stale_results["stale_count"]

    if total_errors > 0:
        logger.error(f"Found {total_errors} critical issues")
        return 1
    elif args.fail_on_warnings and total_warnings > 0:
        logger.warning(
            f"Found {total_warnings} warnings and --fail-on-warnings is set"
        )
        return 1
    else:
        logger.info("All consistency checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
