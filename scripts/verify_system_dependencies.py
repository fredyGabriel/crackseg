#!/usr/bin/env python3
"""
System Dependencies Verification - CrackSeg Project

This script verifies that all system dependencies are installed
and correctly configured for the CrackSeg project.

Usage:
    python scripts/verify_system_dependencies.py
"""

import subprocess
import sys
from collections.abc import Callable


def run_command(
    command: str, capture_output: bool = True
) -> tuple[int, str, str]:
    """
    Execute a system command and return exit code and output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_command_available(
    command: str, version_flag: str = "--version"
) -> tuple[bool, str]:
    """Check if a command is available in the system."""
    full_command = f"{command} {version_flag}"
    returncode, stdout, stderr = run_command(full_command)

    if returncode == 0:
        return True, stdout
    else:
        return False, stderr


def check_python_import(module_name: str) -> tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True, f"[OK] Module '{module_name}' imported successfully"
    except ImportError as e:
        return False, f"[ERROR] Error importing '{module_name}': {e}"


def check_graphviz_integration() -> tuple[bool, str]:
    """Check complete Graphviz integration with Python."""
    try:
        import graphviz

        # Create a simple graph to test functionality
        dot = graphviz.Digraph(comment="Test")
        dot.node("A", "Test Node")

        # Verify that source can be generated
        source = dot.source
        if "Test Node" in source:
            return True, "[OK] Graphviz Python integration working"
        else:
            return (
                False,
                "[ERROR] Graphviz Python does not generate correct output",
            )
    except Exception as e:
        return False, f"[ERROR] Error in Graphviz integration: {e}"


def check_torch_cuda() -> tuple[bool, str]:
    """Check CUDA availability with PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            )
            # Get CUDA version safely
            try:
                cuda_version = torch.version.cuda  # type: ignore
            except AttributeError:
                cuda_version = "N/A"

            return (
                True,
                f"[OK] CUDA available: {device_count} GPU(s), "
                f"{device_name}, CUDA {cuda_version}",
            )
        else:
            return False, "[WARN] CUDA not available (using CPU)"
    except Exception as e:
        return False, f"[ERROR] Error checking CUDA: {e}"


def check_environment_setup() -> tuple[bool, str]:
    """Check conda environment configuration."""
    # Verify we are in the correct environment
    conda_env = run_command("conda info --envs | grep '*'")[1]

    if "crackseg" in conda_env:
        return True, "[OK] Conda environment 'crackseg' active"
    else:
        return (
            False,
            f"[WARN] Current environment: {conda_env.strip()}, "
            f"expected: crackseg",
        )


def main() -> None:
    """Main verification function."""
    print("System Dependencies Verification - CrackSeg Project")
    print("=" * 70)

    # List of checks - use Callable from collections.abc
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("Git", lambda: check_command_available("git")),
        ("Conda", lambda: check_command_available("conda")),
        ("Python", lambda: check_command_available("python")),
        ("Pip", lambda: check_command_available("pip")),
        ("Graphviz (command)", lambda: check_command_available("dot", "-V")),
        ("Conda Environment", check_environment_setup),
        ("Graphviz (Python)", lambda: check_python_import("graphviz")),
        ("Graphviz (integration)", check_graphviz_integration),
        ("PyTorch", lambda: check_python_import("torch")),
        ("CUDA Support", check_torch_cuda),
        ("Streamlit", lambda: check_python_import("streamlit")),
        ("OpenCV", lambda: check_python_import("cv2")),
        ("NumPy", lambda: check_python_import("numpy")),
        ("Hydra", lambda: check_python_import("hydra")),
    ]

    # Execute checks
    results: dict[str, tuple[bool, str]] = {}
    all_passed = True

    for check_name, check_function in checks:
        print(f"\nChecking {check_name}...")
        try:
            success, message = check_function()
            results[check_name] = (success, message)
            print(f"   {message}")

            if not success and check_name in [
                "Git",
                "Conda",
                "Python",
                "Graphviz (command)",
            ]:
                all_passed = False
        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {e}"
            results[check_name] = (False, error_msg)
            print(f"   {error_msg}")
            all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    required_checks = [
        "Git",
        "Conda",
        "Python",
        "Graphviz (command)",
        "Conda Environment",
    ]
    optional_checks = ["CUDA Support"]

    print("\nREQUIRED DEPENDENCIES:")
    for check_name in required_checks:
        if check_name in results:
            success, message = results[check_name]
            status = "[PASS]" if success else "[FAIL]"
            print(f"   {status} {check_name}")

    print("\nOPTIONAL DEPENDENCIES:")
    for check_name in optional_checks:
        if check_name in results:
            success, message = results[check_name]
            status = "[AVAILABLE]" if success else "[NOT AVAILABLE]"
            print(f"   {status} {check_name}")

    print("\nPYTHON DEPENDENCIES:")
    python_checks = [
        name
        for name in results.keys()
        if name not in required_checks + optional_checks
    ]
    for check_name in python_checks:
        success, message = results[check_name]
        status = "[OK]" if success else "[FAIL]"
        print(f"   {status} {check_name}")

    # Recommendations
    print("\nRECOMMENDATIONS:")

    failed_required = [
        name
        for name in required_checks
        if name in results and not results[name][0]
    ]

    if failed_required:
        print("[ERROR] The following required dependencies are missing:")
        for name in failed_required:
            print(f"   - {name}")
        print(
            "\n[INFO] Check docs/guides/SYSTEM_DEPENDENCIES.md "
            "for installation instructions"
        )

    if "CUDA Support" in results and not results["CUDA Support"][0]:
        print("[WARN] CUDA not available - Training will be slower using CPU")
        print("[INFO] For better performance, install compatible CUDA Toolkit")

    if all_passed and not failed_required:
        print("[SUCCESS] All required dependencies are correctly installed!")
        print("[SUCCESS] The project is ready to use")
    else:
        print("[WARN] Some dependencies require attention before continuing")

    print("\n" + "=" * 70)

    # Exit code
    sys.exit(0 if not failed_required else 1)


if __name__ == "__main__":
    main()
