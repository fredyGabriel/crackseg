"""Import policy checker for enforcing re-exports and import standards.

This script checks for violations of the project's import policy, including:
- Direct imports from internal modules that should use re-exports
- Missing re-exports for public APIs
- Circular import dependencies
- Import organization violations
"""

import argparse
import ast
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

from simple_mapping_registry import (  # noqa: E402
    SimpleMappingRegistry,
    create_default_registry,
)

from scripts.utils.common.io_utils import read_text  # noqa: E402
from scripts.utils.common.logging_utils import setup_logging  # noqa: E402


class ImportPolicyChecker:
    """Checker for import policy violations."""

    def __init__(self, registry: SimpleMappingRegistry):
        """Initialize the checker.

        Args:
            registry: Mapping registry for path validation
        """
        self.registry = registry
        self.violations = []

        # Define public API modules that should be re-exported
        self.public_api_modules = {
            "crackseg.model",
            "crackseg.training",
            "crackseg.evaluation",
            "crackseg.data",
            "crackseg.utils",
            "crackseg.reporting",
        }

        # Define internal modules that should not be imported directly
        self.internal_modules = {
            "crackseg.model.encoder",
            "crackseg.model.decoder",
            "crackseg.model.bottleneck",
            "crackseg.training.trainer",
            "crackseg.training.loss",
            "crackseg.utils.core",
            "crackseg.utils.logging",
            "crackseg.utils.config",
        }

        # Define allowed direct imports (exceptions to the rule)
        self.allowed_direct_imports = {
            "crackseg.utils.core.CrackSegError",
            "crackseg.utils.core.get_device",
            "crackseg.utils.core.set_random_seeds",
            "crackseg.utils.config.ConfigSchema",
            "crackseg.utils.config.validate_config",
        }

    def check_file_imports(self, file_path: Path) -> list[dict]:
        """Check imports in a single Python file.

        Args:
            file_path: Path to the Python file to check

        Returns:
            List of import policy violations found
        """
        violations = []

        try:
            content = read_text(file_path)

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        violation = self._check_import_statement(
                            alias.name, file_path, "import"
                        )
                        if violation:
                            violations.append(violation)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_name = (
                            f"{module}.{alias.name}" if module else alias.name
                        )
                        violation = self._check_import_statement(
                            full_name, file_path, "from"
                        )
                        if violation:
                            violations.append(violation)

            return violations

        except Exception as e:
            return [
                {
                    "file": str(file_path),
                    "issue": "parse_error",
                    "error": str(e),
                    "severity": "error",
                }
            ]

    def _check_import_statement(
        self, import_name: str, file_path: Path, import_type: str
    ) -> dict:
        """Check a single import statement for policy violations.

        Args:
            import_name: Name of the imported module/object
            file_path: Path to the file containing the import
            import_type: Type of import ("import" or "from")

        Returns:
            Violation dict if found, None otherwise
        """
        # Skip relative imports
        if import_name.startswith("."):
            return None

        # Check if this is an allowed direct import
        if import_name in self.allowed_direct_imports:
            return None

        # Check for internal module violations
        for internal_module in self.internal_modules:
            if import_name.startswith(internal_module):
                return {
                    "file": str(file_path),
                    "import_name": import_name,
                    "import_type": import_type,
                    "issue": "internal_module_direct_import",
                    "severity": "error",
                    "message": f"Direct import from internal module '{internal_module}' should use re-export",
                }

        # Check for missing re-exports in public API
        if import_name.startswith("crackseg."):
            parts = import_name.split(".")
            if len(parts) >= 2:
                public_module = f"{parts[0]}.{parts[1]}"
                if public_module in self.public_api_modules:
                    # This should be available through re-export
                    return {
                        "file": str(file_path),
                        "import_name": import_name,
                        "import_type": import_type,
                        "issue": "missing_re_export",
                        "severity": "warning",
                        "message": f"Import '{import_name}' should be available through re-export from '{public_module}'",
                    }

        return None

    def check_re_exports(self, module_path: Path) -> list[dict]:
        """Check if a module properly re-exports its public API.

        Args:
            module_path: Path to the module to check

        Returns:
            List of re-export violations found
        """
        violations = []

        # Check __init__.py files for proper re-exports
        init_file = module_path / "__init__.py"
        if not init_file.exists():
            return violations

        try:
            content = read_text(init_file)

            tree = ast.parse(content)

            # Check for __all__ definition
            has_all = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "__all__"
                        ):
                            has_all = True
                            break

            if not has_all:
                violations.append(
                    {
                        "file": str(init_file),
                        "issue": "missing_all_definition",
                        "severity": "warning",
                        "message": "Module should define __all__ for explicit public API",
                    }
                )

            # Check for re-export patterns
            re_exports_found = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("."):
                        # Relative import - likely a re-export
                        for alias in node.names:
                            re_exports_found.append(alias.name)

            if not re_exports_found:
                violations.append(
                    {
                        "file": str(init_file),
                        "issue": "no_re_exports",
                        "severity": "warning",
                        "message": "Module should re-export public API from submodules",
                    }
                )

            return violations

        except Exception as e:
            return [
                {
                    "file": str(init_file),
                    "issue": "parse_error",
                    "error": str(e),
                    "severity": "error",
                }
            ]

    def check_circular_imports(self, src_dir: Path) -> list[dict]:
        """Check for circular import dependencies.

        Args:
            src_dir: Source directory to check

        Returns:
            List of circular import violations found
        """
        # This is a simplified check - a full circular import detector
        # would need to build a dependency graph and detect cycles
        violations = []

        # For now, we'll check for obvious circular patterns
        # like module A importing from module B and vice versa
        import_graph = {}

        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            module_name = self._get_module_name(py_file, src_dir)
            imports = self._get_file_imports(py_file)

            if module_name not in import_graph:
                import_graph[module_name] = set()

            for imp in imports:
                if imp.startswith("crackseg."):
                    import_graph[module_name].add(imp)

        # Check for obvious circular dependencies
        for module, imports in import_graph.items():
            for imported_module in imports:
                if imported_module in import_graph:
                    if module in import_graph[imported_module]:
                        violations.append(
                            {
                                "file": f"circular_import_{module}_{imported_module}",
                                "issue": "circular_import",
                                "severity": "error",
                                "message": f"Circular import detected: {module} <-> {imported_module}",
                            }
                        )

        return violations

    def _get_module_name(self, file_path: Path, src_dir: Path) -> str:
        """Get the module name for a file path.

        Args:
            file_path: Path to the file
            src_dir: Source directory root

        Returns:
            Module name
        """
        relative_path = file_path.relative_to(src_dir)
        parts = relative_path.parts
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts = parts[:-1] + (parts[-1].replace(".py", ""),)

        return ".".join(parts)

    def _get_file_imports(self, file_path: Path) -> list[str]:
        """Get all imports from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of imported module names
        """
        imports = []

        try:
            content = read_text(file_path)

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports

        except Exception:
            return []


def check_directory_imports(
    directory: Path, checker: ImportPolicyChecker
) -> list[dict]:
    """Check imports in all Python files in a directory.

    Args:
        directory: Directory to check
        checker: Import policy checker instance

    Returns:
        List of all import policy violations found
    """
    all_violations = []

    for py_file in directory.rglob("*.py"):
        if not py_file.is_file():
            continue

        # Skip test files for now (could be configurable)
        if "test" in py_file.name.lower():
            continue

        file_violations = checker.check_file_imports(py_file)
        all_violations.extend(file_violations)

        # Check re-exports for __init__.py files
        if py_file.name == "__init__.py":
            re_export_violations = checker.check_re_exports(py_file.parent)
            all_violations.extend(re_export_violations)

    return all_violations


def generate_import_report(violations: list[dict]) -> str:
    """Generate a human-readable report from import violations.

    Args:
        violations: List of import policy violations found

    Returns:
        Formatted report string
    """
    if not violations:
        return "✅ No import policy violations found!"

    report_lines = ["📦 Import Policy Checker Report", "=" * 50, ""]

    # Group by severity
    errors = [v for v in violations if v.get("severity") == "error"]
    warnings = [v for v in violations if v.get("severity") == "warning"]

    if errors:
        report_lines.append("❌ ERRORS:")
        for violation in errors:
            if violation.get("issue") == "parse_error":
                report_lines.append(
                    f"  - {violation['file']}: {violation['error']}"
                )
            elif violation.get("issue") == "internal_module_direct_import":
                report_lines.append(
                    f"  - {violation['file']}: {violation['message']} "
                    f"(import: {violation['import_name']})"
                )
            elif violation.get("issue") == "circular_import":
                report_lines.append(f"  - {violation['message']}")
        report_lines.append("")

    if warnings:
        report_lines.append("⚠️  WARNINGS:")
        for violation in warnings:
            if violation.get("issue") == "missing_re_export":
                report_lines.append(
                    f"  - {violation['file']}: {violation['message']} "
                    f"(import: {violation['import_name']})"
                )
            elif violation.get("issue") == "missing_all_definition":
                report_lines.append(
                    f"  - {violation['file']}: {violation['message']}"
                )
            elif violation.get("issue") == "no_re_exports":
                report_lines.append(
                    f"  - {violation['file']}: {violation['message']}"
                )
        report_lines.append("")

    # Summary
    report_lines.append(
        f"Summary: {len(errors)} errors, {len(warnings)} warnings"
    )

    return "\n".join(report_lines)


def main() -> int:
    """Main function to run the import policy checker.

    Returns:
        Exit code (0 for success, 1 for violations found)
    """
    parser = argparse.ArgumentParser(
        description="Check import policy violations"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["src"],
        help="Directories to check for import violations",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Get registry
    registry = create_default_registry()

    # Create checker
    checker = ImportPolicyChecker(registry)

    # Check directories
    all_violations = []

    for directory_str in args.directories:
        directory = Path(directory_str)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        logger.info(f"Checking imports in: {directory}")
        directory_violations = check_directory_imports(directory, checker)
        all_violations.extend(directory_violations)
        logger.info(f"  Found {len(directory_violations)} violations")

    # Check for circular imports
    logger.info("Checking for circular imports...")
    circular_violations = checker.check_circular_imports(Path("src"))
    all_violations.extend(circular_violations)
    logger.info(
        f"  Found {len(circular_violations)} circular import violations"
    )

    # Generate and print report
    report = generate_import_report(all_violations)
    print(report)

    # Return appropriate exit code
    errors = [v for v in all_violations if v.get("severity") == "error"]
    warnings = [v for v in all_violations if v.get("severity") == "warning"]

    if errors:
        logger.error(f"Found {len(errors)} errors")
        return 1
    elif warnings:
        logger.warning(f"Found {len(warnings)} warnings")
        return 0  # Warnings don't fail CI
    else:
        logger.info("No violations found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
