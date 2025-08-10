"""Hydra Config Migration Helper and Consistency Validator.

This script helps migrate Hydra configuration files when project structure
changes and validates their consistency with the mapping registry.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

from simple_mapping_registry import (  # noqa: E402
    SimpleMappingRegistry,
    create_default_registry,
)

from scripts.utils.common.logging_utils import setup_logging  # noqa: E402


class HydraConfigMigrator:
    """Helper for migrating and validating Hydra configuration files."""

    def __init__(self, registry: SimpleMappingRegistry):
        """Initialize the migrator.

        Args:
            registry: Mapping registry for path validation
        """
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def parse_hydra_config(self, content: str) -> dict[str, Any]:
        """Parse a Hydra configuration file.

        Args:
            content: Configuration file content

        Returns:
            Dictionary with parsed configuration
        """
        config = {}
        current_section = None

        for _line_num, line in enumerate(content.split("\n"), 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Section header
            if line.endswith(":") and not line.startswith(" "):
                current_section = line[:-1]
                config[current_section] = {}
                continue

            # Key-value pair
            if ":" in line and current_section:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()

                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                config[current_section][key] = value

        return config

    def find_config_references(self, content: str) -> list[dict[str, Any]]:
        """Find configuration references in Hydra config content.

        Args:
            content: Configuration file content

        Returns:
            List of configuration references found
        """
        references = []

        # Pattern for Hydra config references
        patterns = [
            # Defaults section
            r"defaults:\s*\n\s*-\s*([^\s]+)",
            # Direct config references
            r"config_path:\s*([^\s]+)",
            r"config_name:\s*([^\s]+)",
            # Import statements
            r"import\s+([^\s]+)",
            # Package references
            r"package:\s*([^\s]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                ref_path = match.group(1)
                references.append(
                    {
                        "path": ref_path,
                        "pattern": pattern,
                        "match": match.group(0),
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        return references

    def validate_config_references(
        self, config_path: Path, content: str
    ) -> list[dict[str, Any]]:
        """Validate configuration references against the mapping registry.

        Args:
            config_path: Path to the configuration file
            content: Configuration file content

        Returns:
            List of validation issues found
        """
        issues = []
        references = self.find_config_references(content)

        for ref in references:
            ref_path = ref["path"]

            # Check if this reference needs migration
            for mapping in self.registry.mappings:
                if (
                    mapping.mapping_type == "config"
                    and mapping.old_path in ref_path
                ):
                    # Check if the new path exists
                    new_path = ref_path.replace(
                        mapping.old_path, mapping.new_path
                    )

                    # Resolve the path relative to the config file
                    if new_path.startswith("/"):
                        resolved_path = project_root / new_path[1:]
                    else:
                        resolved_path = config_path.parent / new_path

                    if resolved_path.exists():
                        issues.append(
                            {
                                "type": "migration_needed",
                                "severity": "warning",
                                "line": ref["line"],
                                "old_path": ref_path,
                                "new_path": new_path,
                                "mapping": f"{mapping.old_path} -> {mapping.new_path}",
                                "description": f"Config reference can be migrated to {new_path}",
                            }
                        )
                    else:
                        issues.append(
                            {
                                "type": "broken_reference",
                                "severity": "error",
                                "line": ref["line"],
                                "old_path": ref_path,
                                "new_path": new_path,
                                "mapping": f"{mapping.old_path} -> {mapping.new_path}",
                                "description": f"Config reference {ref_path} is broken",
                            }
                        )
                    break
            else:
                # Check if the reference exists
                if ref_path.startswith("/"):
                    resolved_path = project_root / ref_path[1:]
                else:
                    resolved_path = config_path.parent / ref_path

                if not resolved_path.exists():
                    issues.append(
                        {
                            "type": "missing_reference",
                            "severity": "error",
                            "line": ref["line"],
                            "old_path": ref_path,
                            "description": f"Config reference {ref_path} does not exist",
                        }
                    )

        return issues

    def migrate_config_file(
        self, config_path: Path, dry_run: bool = True
    ) -> dict[str, Any]:
        """Migrate a single Hydra configuration file.

        Args:
            config_path: Path to the configuration file
            dry_run: If True, only preview changes

        Returns:
            Dictionary with migration results
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                content = f.read()

            # Find all config references
            references = self.find_config_references(content)
            migrated_count = 0
            errors = 0

            # Sort references by line number in descending order to avoid line shifts
            sorted_refs = sorted(
                references, key=lambda x: x["line"], reverse=True
            )

            lines = content.split("\n")

            for ref in sorted_refs:
                ref_path = ref["path"]
                line_num = ref["line"] - 1  # Convert to 0-based index

                # Find the best mapping for this reference
                best_mapping = None
                for mapping in self.registry.mappings:
                    if (
                        mapping.mapping_type == "config"
                        and mapping.old_path in ref_path
                    ):
                        new_path = ref_path.replace(
                            mapping.old_path, mapping.new_path
                        )

                        # Check if the new path exists
                        if new_path.startswith("/"):
                            resolved_path = project_root / new_path[1:]
                        else:
                            resolved_path = config_path.parent / new_path

                        if resolved_path.exists():
                            best_mapping = mapping
                            break

                if best_mapping and line_num < len(lines):
                    old_line = lines[line_num]
                    new_path = ref_path.replace(
                        best_mapping.old_path, best_mapping.new_path
                    )

                    # Replace the reference in the line
                    new_line = old_line.replace(ref_path, new_path)

                    if new_line != old_line:
                        lines[line_num] = new_line
                        migrated_count += 1

                        if not dry_run:
                            self.logger.info(
                                f"Migrated {config_path}:{line_num + 1} "
                                f"'{ref_path}' -> '{new_path}'"
                            )
                        else:
                            self.logger.info(
                                f"Would migrate {config_path}:{line_num + 1} "
                                f"'{ref_path}' -> '{new_path}'"
                            )
                    else:
                        errors += 1
                        self.logger.warning(
                            f"Could not migrate {config_path}:{line_num + 1} "
                            f"'{ref_path}' -> '{new_path}'"
                        )
                else:
                    errors += 1
                    self.logger.error(
                        f"Line {line_num + 1} not found in {config_path}"
                    )

            # Write the migrated content
            if not dry_run and migrated_count > 0:
                new_content = "\n".join(lines)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

            return {
                "migrated": migrated_count,
                "errors": errors,
                "total_references": len(references),
            }

        except Exception as e:
            self.logger.error(f"Error migrating {config_path}: {e}")
            return {"migrated": 0, "errors": 1, "total_references": 0}

    def scan_config_directory(self, directory: Path) -> list[Path]:
        """Scan a directory for Hydra configuration files.

        Args:
            directory: Directory to scan

        Returns:
            List of configuration file paths
        """
        config_files = []

        # Common Hydra config extensions
        config_extensions = {".yaml", ".yml", ".json"}

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in config_extensions:
                config_files.append(file_path)

        return config_files

    def validate_config_directory(
        self, directory: Path
    ) -> list[dict[str, Any]]:
        """Validate all configuration files in a directory.

        Args:
            directory: Directory to validate

        Returns:
            List of all validation issues found
        """
        all_issues = []
        config_files = self.scan_config_directory(directory)

        for config_file in config_files:
            try:
                with open(config_file, encoding="utf-8") as f:
                    content = f.read()

                file_issues = self.validate_config_references(
                    config_file, content
                )

                for issue in file_issues:
                    issue["file"] = str(config_file)

                all_issues.extend(file_issues)

            except Exception as e:
                all_issues.append(
                    {
                        "file": str(config_file),
                        "type": "file_error",
                        "severity": "error",
                        "description": f"Error reading file: {e}",
                    }
                )

        return all_issues

    def migrate_config_directory(
        self, directory: Path, dry_run: bool = True
    ) -> dict[str, Any]:
        """Migrate all configuration files in a directory.

        Args:
            directory: Directory to migrate
            dry_run: If True, only preview changes

        Returns:
            Dictionary with migration results
        """
        config_files = self.scan_config_directory(directory)
        total_migrated = 0
        total_errors = 0
        total_references = 0

        for config_file in config_files:
            self.logger.info(f"Processing {config_file}...")
            result = self.migrate_config_file(config_file, dry_run)
            total_migrated += result["migrated"]
            total_errors += result["errors"]
            total_references += result["total_references"]

        return {
            "files_processed": len(config_files),
            "total_migrated": total_migrated,
            "total_errors": total_errors,
            "total_references": total_references,
        }


def generate_validation_report(issues: list[dict[str, Any]]) -> str:
    """Generate a validation report from issues found.

    Args:
        issues: List of validation issues

    Returns:
        Formatted report string
    """
    if not issues:
        return "✅ No validation issues found!"

    report_lines = [
        "🔧 Hydra Config Validation Report",
        "=" * 50,
        "",
    ]

    # Group by severity
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    # Summary
    report_lines.append("📊 SUMMARY:")
    report_lines.append(f"  - Total Issues: {len(issues)}")
    report_lines.append(f"  - Errors: {len(errors)}")
    report_lines.append(f"  - Warnings: {len(warnings)}")
    report_lines.append("")

    # Group by file
    files = {}
    for issue in issues:
        file_path = issue.get("file", "unknown")
        if file_path not in files:
            files[file_path] = []
        files[file_path].append(issue)

    # Detailed breakdown
    report_lines.append("📁 DETAILED BREAKDOWN:")
    for file_path, file_issues in files.items():
        report_lines.append(f"  {file_path}:")
        for issue in file_issues:
            severity_icon = "❌" if issue.get("severity") == "error" else "⚠️"
            line_info = f" (line {issue['line']})" if "line" in issue else ""
            report_lines.append(
                f"    {severity_icon} {issue['type']}{line_info}: {issue['description']}"
            )
        report_lines.append("")

    return "\n".join(report_lines)


def generate_migration_report(results: dict[str, Any]) -> str:
    """Generate a migration report from results.

    Args:
        results: Migration results

    Returns:
        Formatted report string
    """
    report_lines = [
        "🔄 Hydra Config Migration Report",
        "=" * 50,
        "",
    ]

    # Summary
    report_lines.append("📊 SUMMARY:")
    report_lines.append(f"  - Files Processed: {results['files_processed']}")
    report_lines.append(
        f"  - References Migrated: {results['total_migrated']}"
    )
    report_lines.append(f"  - Errors: {results['total_errors']}")
    report_lines.append(f"  - Total References: {results['total_references']}")
    report_lines.append("")

    # Success rate
    if results["total_references"] > 0:
        success_rate = (
            results["total_migrated"] / results["total_references"] * 100
        )
        report_lines.append(f"  - Success Rate: {success_rate:.1f}%")
    else:
        report_lines.append("  - No references found to migrate")

    return "\n".join(report_lines)


def main() -> int:
    """Main function to run the Hydra config migrator.

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description="Migrate and validate Hydra configuration files"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["configs"],
        help="Directories to scan for config files",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configs, don't migrate",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate config files (default is dry-run)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Get registry
    registry = create_default_registry()

    # Create migrator
    migrator = HydraConfigMigrator(registry)

    # Determine if this is a dry run
    dry_run = not args.migrate

    if args.validate_only:
        logger.info("🔍 Running validation only...")

        # Validate all directories
        all_issues = []
        for directory_str in args.directories:
            directory = Path(directory_str)
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                continue

            logger.info(f"Validating {directory}...")
            directory_issues = migrator.validate_config_directory(directory)
            all_issues.extend(directory_issues)
            logger.info(f"  Found {len(directory_issues)} issues")

        # Generate and print report
        report = generate_validation_report(all_issues)
        print(report)

        # Return appropriate exit code
        errors = [i for i in all_issues if i.get("severity") == "error"]
        if errors:
            logger.error(f"Found {len(errors)} validation errors")
            return 1
        else:
            logger.info("Validation completed successfully")
            return 0

    else:
        if dry_run:
            logger.info(
                "🔍 Running in DRY-RUN mode (no changes will be applied)"
            )
        else:
            logger.info("🔧 Running in MIGRATE mode (changes will be applied)")

        # Migrate all directories
        total_results = {
            "files_processed": 0,
            "total_migrated": 0,
            "total_errors": 0,
            "total_references": 0,
        }

        for directory_str in args.directories:
            directory = Path(directory_str)
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                continue

            logger.info(f"Migrating {directory}...")
            directory_results = migrator.migrate_config_directory(
                directory, dry_run
            )

            total_results["files_processed"] += directory_results[
                "files_processed"
            ]
            total_results["total_migrated"] += directory_results[
                "total_migrated"
            ]
            total_results["total_errors"] += directory_results["total_errors"]
            total_results["total_references"] += directory_results[
                "total_references"
            ]

        # Generate and print report
        report = generate_migration_report(total_results)
        print(report)

        # Return appropriate exit code
        if total_results["total_errors"] > 0:
            logger.error(
                f"Encountered {total_results['total_errors']} errors during migration"
            )
            return 1
        elif total_results["total_migrated"] > 0:
            if dry_run:
                logger.info(
                    f"Would migrate {total_results['total_migrated']} references (run with --migrate to apply)"
                )
            else:
                logger.info(
                    f"Successfully migrated {total_results['total_migrated']} references"
                )
            return 0
        else:
            logger.info("No references to migrate")
            return 0


if __name__ == "__main__":
    sys.exit(main())
