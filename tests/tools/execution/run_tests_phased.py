#!/usr/bin/env python3
"""
Phased Test Execution Script for CrackSeg Project This script executes
tests in phases to avoid dependency issues and provide controlled
testing of the entire project.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class TestPhase:
    """Represents a test execution phase."""

    def __init__(
        self, name: str, command: str, description: str, critical: bool = False
    ):
        self.name = name
        self.command = command
        self.description = description
        self.critical = critical
        self.status = "pending"
        self.duration = 0.0
        self.error: str | None = None


class TestExecutor:
    """Manages phased test execution."""

    def __init__(self):
        self.phases: list[TestPhase] = []
        self.results: dict[str, Any] = {}
        self._setup_phases()

    def _setup_phases(self):
        """Setup test phases in execution order."""
        self.phases = [
            TestPhase(
                "utils",
                "python -m pytest tests/unit/utils/ -v --tb=short",
                "Basic utility tests",
                critical=True,
            ),
            TestPhase(
                "config",
                "python -m pytest tests/integration/config/ -v --tb=short",
                "Configuration system tests",
                critical=True,
            ),
            TestPhase(
                "tools",
                "python -m pytest tests/tools/ -v --tb=short",
                "Testing tools and utilities",
                critical=False,
            ),
            TestPhase(
                "data_unit",
                "python -m pytest tests/unit/data/ -v --tb=short",
                "Data pipeline unit tests",
                critical=True,
            ),
            TestPhase(
                "data_integration",
                "python -m pytest tests/integration/data/ -v --tb=short",
                "Data pipeline integration tests",
                critical=True,
            ),
            TestPhase(
                "evaluation",
                "python -m pytest tests/unit/evaluation/ "
                "tests/integration/evaluation/ -v --tb=short",
                "Evaluation metrics tests",
                critical=True,
            ),
            TestPhase(
                "model_unit",
                "python -m pytest tests/unit/model/ -v --tb=short",
                "Model architecture unit tests",
                critical=True,
            ),
            TestPhase(
                "model_integration",
                "python -m pytest tests/integration/model/ -v --tb=short",
                "Model integration tests",
                critical=True,
            ),
            TestPhase(
                "training_unit",
                "python -m pytest tests/unit/training/ -v --tb=short",
                "Training pipeline unit tests",
                critical=True,
            ),
            TestPhase(
                "training_integration",
                "python -m pytest tests/integration/training/ -v --tb=short",
                "Training integration tests",
                critical=True,
            ),
            TestPhase(
                "gui_unit",
                "python -m pytest tests/gui/ -v --tb=short",
                "GUI component tests",
                critical=False,
            ),
            TestPhase(
                "gui_integration",
                "python -m pytest tests/integration/gui/ -v --tb=short",
                "GUI integration tests",
                critical=False,
            ),
        ]

    def run_phase(self, phase: TestPhase) -> bool:
        """Execute a single test phase."""
        print(f"\n{'=' * 60}")
        print(f"Phase: {phase.name}")
        print(f"Description: {phase.description}")
        print(f"Critical: {'Yes' if phase.critical else 'No'}")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            # Ensure conda environment is activated
            full_command = f"conda activate crackseg && {phase.command}"

            print(f"Executing: {phase.command}")
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per phase
            )

            phase.duration = time.time() - start_time

            if result.returncode == 0:
                phase.status = "passed"
                print(f"✅ Phase {phase.name} PASSED ({phase.duration:.2f}s)")
                return True
            else:
                # Check if this is a torchvision/PIL compatibility issue
                if (
                    "DLL load failed" in result.stderr
                    or "0xc0000138" in result.stderr
                ):
                    print(
                        f"🔧 Detected torchvision/PIL compatibility issue "
                        f"in phase {phase.name}"
                    )
                    print("Attempting to resolve the issue...")

                    # Try to fix the compatibility issue
                    if self._fix_torchvision_compatibility():
                        print(
                            f"🔄 Retrying phase {phase.name} "
                            f"after compatibility fix..."
                        )
                        # Retry the phase
                        retry_result = subprocess.run(
                            full_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )

                        if retry_result.returncode == 0:
                            phase.status = "passed"
                            phase.duration = time.time() - start_time
                            print(
                                f"✅ Phase {phase.name} PASSED after fix "
                                f"({phase.duration:.2f}s)"
                            )
                            return True
                        else:
                            phase.status = "failed"
                            phase.error = retry_result.stderr
                            print(
                                f"❌ Phase {phase.name} FAILED after fix "
                                f"({phase.duration:.2f}s)"
                            )
                            print(f"Error: {retry_result.stderr}")
                            return False
                    else:
                        phase.status = "failed"
                        phase.error = result.stderr
                        print(
                            f"❌ Phase {phase.name} FAILED - "
                            f"could not fix compatibility issue"
                        )
                        return False
                else:
                    phase.status = "failed"
                    phase.error = result.stderr
                    print(
                        f"❌ Phase {phase.name} FAILED ({phase.duration:.2f}s)"
                    )
                    print(f"Error: {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            phase.status = "timeout"
            phase.duration = time.time() - start_time
            print(f"⏰ Phase {phase.name} TIMEOUT ({phase.duration:.2f}s)")
            return False
        except Exception as e:
            phase.status = "error"
            phase.duration = time.time() - start_time
            phase.error = str(e)
            print(f"💥 Phase {phase.name} ERROR ({phase.duration:.2f}s): {e}")
            return False

    def _fix_torchvision_compatibility(self) -> bool:
        """Attempt to fix torchvision/PIL compatibility issues."""
        try:
            print("  - Running torchvision compatibility fix...")
            fix_script = (
                Path(__file__).parent / "fix_torchvision_compatibility.py"
            )

            if not fix_script.exists():
                print(
                    "  - Compatibility fix script not found, "
                    "creating basic fix..."
                )
                return self._basic_compatibility_fix()

            result = subprocess.run(
                f"conda activate crackseg && python {fix_script}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout for fix
            )

            if result.returncode == 0:
                print(
                    "  ✅ TorchVision compatibility fix applied successfully"
                )
                return True
            else:
                print(
                    f"  ❌ TorchVision compatibility fix failed: "
                    f"{result.stderr}"
                )
                return False

        except Exception as e:
            print(f"  ❌ Error during compatibility fix: {e}")
            return False

    def _basic_compatibility_fix(self) -> bool:
        """
        Apply basic compatibility fix when the full script is not available.
        """
        try:
            print("  - Applying basic compatibility fix...")

            # Try to reinstall Pillow with a compatible version
            result = subprocess.run(
                "conda activate crackseg && pip uninstall pillow -y "
                "&& pip install pillow==11.3.0",
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                print("  ✅ Basic compatibility fix applied")
                return True
            else:
                print(f"  ❌ Basic compatibility fix failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"  ❌ Error during basic compatibility fix: {e}")
            return False

    def run_all_phases(
        self, stop_on_critical_failure: bool = True
    ) -> dict[str, Any]:
        """Execute all test phases."""
        print("🚀 Starting Phased Test Execution")
        print(f"Total phases: {len(self.phases)}")

        passed = 0
        failed = 0
        critical_failures = 0

        for i, phase in enumerate(self.phases, 1):
            print(f"\n[{i}/{len(self.phases)}] Starting phase: {phase.name}")

            success = self.run_phase(phase)

            if success:
                passed += 1
            else:
                failed += 1
                if phase.critical:
                    critical_failures += 1
                    if stop_on_critical_failure:
                        print(
                            f"\n🛑 Critical failure in phase {phase.name}. "
                            "Stopping execution."
                        )
                        break

        # Generate summary
        self.results = {
            "total_phases": len(self.phases),
            "passed": passed,
            "failed": failed,
            "critical_failures": critical_failures,
            "success_rate": (passed / len(self.phases)) * 100,
            "phases": {
                phase.name: {
                    "status": phase.status,
                    "duration": phase.duration,
                    "error": phase.error,
                    "critical": phase.critical,
                }
                for phase in self.phases
            },
        }

        return self.results

    def print_summary(self):
        """Print execution summary."""
        print(f"\n{'=' * 60}")
        print("📊 TEST EXECUTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Phases: {self.results['total_phases']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Critical Failures: {self.results['critical_failures']}")
        print(f"Success Rate: {self.results['success_rate']:.1f}%")

        print("\n📋 Phase Details:")
        for phase_name, details in self.results["phases"].items():
            status_icon = "✅" if details["status"] == "passed" else "❌"
            critical_mark = " [CRITICAL]" if details["critical"] else ""
            print(
                f"  {status_icon} {phase_name}{critical_mark}: "
                f"{details['status']} ({details['duration']:.2f}s)"
            )
            if details["error"]:
                print(f"    Error: {details['error'][:100]}...")

        # Recommendations
        if self.results["critical_failures"] > 0:
            print(
                f"\n⚠️  WARNING: {self.results['critical_failures']} "
                "critical failures detected!"
            )
            print(
                "   Critical failures indicate fundamental issues "
                "that must be resolved."
            )
        elif self.results["failed"] > 0:
            print(
                f"\n⚠️  NOTE: {self.results['failed']} "
                "non-critical failures detected."
            )
            print("   Consider reviewing failed phases for improvements.")
        else:
            print("\n🎉 SUCCESS: All phases completed successfully!")

    def run_coverage_analysis(self):
        """Run coverage analysis on successful phases."""
        print(f"\n{'=' * 60}")
        print("📈 COVERAGE ANALYSIS")
        print(f"{'=' * 60}")

        coverage_commands = [
            "python -m pytest tests/unit/ --cov=src "
            "--cov-report=term-missing --cov-report=html",
            "python -m pytest tests/unit/data/ --cov=src.crackseg.data "
            "--cov-report=term-missing",
            "python -m pytest tests/unit/model/ --cov=src.crackseg.model "
            "--cov-report=term-missing",
            "python -m pytest tests/unit/training/ --cov=src.crackseg.training"
            " --cov-report=term-missing",
        ]

        for cmd in coverage_commands:
            print(f"\nExecuting: {cmd}")
            try:
                result = subprocess.run(
                    f"conda activate crackseg && {cmd}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if result.returncode == 0:
                    print("✅ Coverage analysis completed")
                else:
                    print(f"❌ Coverage analysis failed: {result.stderr}")
            except Exception as e:
                print(f"💥 Coverage analysis error: {e}")


def main():
    """Main execution function."""
    executor = TestExecutor()

    # Check if user wants to stop on critical failures
    stop_on_critical = "--continue" not in sys.argv

    # Execute all phases
    results = executor.run_all_phases(
        stop_on_critical_failure=stop_on_critical
    )

    # Print summary
    executor.print_summary()

    # Run coverage analysis if requested
    if "--coverage" in sys.argv:
        executor.run_coverage_analysis()

    # Exit with appropriate code
    if results["critical_failures"] > 0:
        print(
            f"\n🛑 Exiting with error code due to "
            f"{results['critical_failures']} critical failures."
        )
        sys.exit(1)
    elif results["failed"] > 0:
        print(
            f"\n⚠️  Exiting with warning due to "
            f"{results['failed']} non-critical failures."
        )
        sys.exit(2)
    else:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
