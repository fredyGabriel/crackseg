#!/usr/bin/env python3
"""
Test File Coverage Checker - CrackSeg Project

This script ensures that newly added source files have corresponding test
files, helping maintain test coverage discipline during development.
It also validates performance test coverage for critical components.

Usage:
    python scripts/check_test_files.py [source_files...]
    python scripts/check_test_files.py --performance-check
"""

import subprocess
import sys
from pathlib import Path


def safe_print(text: str) -> None:
    """
    Print text with Unicode-safe emoji handling for Windows.

    Args:
        text: Text to print, potentially containing emojis
    """
    # Define emoji replacements for Windows compatibility
    emoji_replacements = {
        "🔍": "[CHECK]",
        "❌": "[ERROR]",
        "✅": "[SUCCESS]",
        "⚠️": "[WARNING]",
        "💡": "[TIP]",
        "🔧": "[MAINTENANCE]",
    }

    # Replace emojis if encoding is problematic
    if sys.stdout.encoding in ["cp1252", "ascii"]:
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)

    print(text)


def check_test_files(source_files: list[str]) -> tuple[bool, list[str]]:
    """
    Check if source files have corresponding test files.

    Args:
        source_files: List of source file paths

    Returns:
        Tuple of (all_have_tests, missing_test_files)
    """
    missing_tests = []

    for source_file in source_files:
        source_path = Path(source_file)

        # Skip __init__.py files and non-Python files
        if source_path.name == "__init__.py" or source_path.suffix != ".py":
            continue

        # Skip if file doesn't exist (could be a deleted file in git)
        if not source_path.exists():
            continue

        # Convert src path to corresponding test path
        if source_path.parts[0] == "src":
            # Remove 'src' and add 'tests/unit' prefix
            test_parts = ["tests", "unit"] + list(source_path.parts[1:])
            test_path = Path(*test_parts)

            # Change filename to test_*.py format
            test_filename = f"test_{test_path.stem}.py"
            test_path = test_path.parent / test_filename

            # Check if test file exists
            if not test_path.exists():
                missing_tests.append(f"{source_file} -> {test_path}")

    return len(missing_tests) == 0, missing_tests


def check_performance_test_coverage() -> tuple[bool, list[str]]:
    """
    Check if critical performance components have appropriate test coverage.

    Returns:
        Tuple of (all_covered, missing_performance_tests)
    """
    missing_performance_tests = []

    # Critical components that must have performance tests
    critical_components = [
        "src/model/architectures",
        "src/training/trainer.py",
        "src/data/dataloader.py",
        "src/utils/monitoring",
    ]

    performance_test_base = Path("tests/e2e/performance")

    for component in critical_components:
        component_path = Path(component)

        if component_path.is_dir():
            # Check if there's a corresponding performance test directory
            perf_test_dir = performance_test_base / component_path.name
            if not perf_test_dir.exists():
                missing_performance_tests.append(
                    f"Performance test directory missing: {perf_test_dir}"
                )
        else:
            # Check if there's a corresponding performance test file
            perf_test_file = (
                performance_test_base
                / f"test_{component_path.stem}_performance.py"
            )
            if not perf_test_file.exists():
                missing_performance_tests.append(
                    f"Performance test file missing: {perf_test_file}"
                )

    # Validate that core benchmarking infrastructure exists
    required_benchmark_files = [
        "tests/e2e/performance/benchmark_suite.py",
        "tests/e2e/performance/benchmark_runner.py",
        "tests/e2e/performance/ci_integration.py",
    ]

    for required_file in required_benchmark_files:
        if not Path(required_file).exists():
            missing_performance_tests.append(
                f"Core benchmark file missing: {required_file}"
            )

    return len(missing_performance_tests) == 0, missing_performance_tests


def validate_performance_system() -> tuple[bool, list[str]]:
    """
    Validate that the performance benchmarking system is properly configured.

    Returns:
        Tuple of (system_valid, validation_errors)
    """
    validation_errors = []

    # Check performance configuration
    config_file = Path("configs/testing/performance_thresholds.yaml")
    if not config_file.exists():
        validation_errors.append(
            f"Performance thresholds config missing: {config_file}"
        )

    # Check CI/CD integration
    ci_file = Path(".github/workflows/performance-ci.yml")
    if not ci_file.exists():
        validation_errors.append(f"Performance CI workflow missing: {ci_file}")

    # Check resource monitoring
    monitor_file = Path("src/utils/monitoring/resource_monitor.py")
    if not monitor_file.exists():
        validation_errors.append(f"Resource monitor missing: {monitor_file}")

    # Check cleanup validation
    cleanup_dir = Path("tests/e2e/cleanup")
    if not cleanup_dir.exists():
        validation_errors.append(
            f"Cleanup validation system missing: {cleanup_dir}"
        )

    # Check maintenance scripts integration
    maintenance_script = Path("scripts/performance_maintenance.py")
    if not maintenance_script.exists():
        validation_errors.append(
            f"Performance maintenance script missing: {maintenance_script}"
        )

    # Check documentation
    docs_file = Path("docs/guides/performance_benchmarking_system.md")
    if not docs_file.exists():
        validation_errors.append(
            f"Performance system documentation missing: {docs_file}"
        )

    return len(validation_errors) == 0, validation_errors


def run_integrated_performance_validation() -> tuple[bool, str]:
    """
    Run comprehensive performance system validation using the maintenance
    script.

    Returns:
        Tuple of (validation_passed, summary_report)
    """
    try:
        # Run the performance maintenance script health check
        result = subprocess.run(
            ["python", "scripts/performance_maintenance.py", "--health-check"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return (
                True,
                f"✅ Performance system health check passed\n{result.stdout}",
            )
        else:
            return (
                False,
                f"❌ Performance system health check failed\n{result.stderr}",
            )

    except subprocess.TimeoutExpired:
        return False, "❌ Performance validation timed out (>2 minutes)"
    except Exception as e:
        return False, f"❌ Performance validation error: {str(e)}"


def main() -> int:
    """Main entry point."""
    # Check for performance system validation flag
    if "--performance-check" in sys.argv:
        safe_print("🔍 Validating performance testing system...")

        # Check performance test coverage
        coverage_ok, missing_perf_tests = check_performance_test_coverage()

        # Validate performance system
        system_ok, validation_errors = validate_performance_system()

        # Run integrated validation with maintenance script
        integration_ok, integration_report = (
            run_integrated_performance_validation()
        )

        # Report results
        all_checks_passed = coverage_ok and system_ok and integration_ok

        if not coverage_ok:
            safe_print("❌ Performance test coverage issues:")
            for missing in missing_perf_tests:
                print(f"  - {missing}")

        if not system_ok:
            safe_print("❌ Performance system validation errors:")
            for error in validation_errors:
                print(f"  - {error}")

        if not integration_ok:
            safe_print("❌ Performance system integration issues:")
            print(f"  {integration_report}")

        if all_checks_passed:
            safe_print("✅ Performance testing system validation passed.")
            safe_print("🔧 System maintenance and integration verified.")
            return 0
        else:
            safe_print("\n⚠️  Performance testing system needs attention.")
            safe_print(
                "💡 Run 'python scripts/performance_maintenance.py "
                "--health-check' for details."
            )
            return 1

    # Regular test file checking
    if len(sys.argv) < 2:
        print("No source files provided to check.")
        return 0

    source_files = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    all_have_tests, missing_tests = check_test_files(source_files)

    if not all_have_tests:
        safe_print("❌ Missing test files for the following source files:")
        for missing in missing_tests:
            print(f"  - {missing}")
        print("\nPlease create corresponding test files before committing.")
        print(
            "Test files should be located in tests/unit/ with 'test_' prefix."
        )
        return 1

    safe_print("✅ All source files have corresponding test files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
