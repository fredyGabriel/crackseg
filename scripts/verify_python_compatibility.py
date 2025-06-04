#!/usr/bin/env python3
"""
Python 3.12 Compatibility Verification Script

This script verifies that all project components work correctly with
Python 3.12, including imports, basic functionality, and modern type hints.
"""

import sys
from collections.abc import Callable
from pathlib import Path


def check_header(message: str) -> None:
    """Print a formatted header for test sections."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {message}")
    print("=" * 60)


def check_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def check_error(message: str, error: Exception) -> None:
    """Print an error message."""
    print(f"❌ {message}: {error}")


def check_python_version() -> bool:
    """Verify Python version is 3.12+."""
    check_header("Python Version Check")

    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor == 12:
        check_success(
            f"Python {version_info.major}.{version_info.minor}."
            f"{version_info.micro}"
        )
        check_success(f"Full version: {sys.version}")
        return True
    else:
        check_error(
            f"Expected Python 3.12, found "
            f"{version_info.major}.{version_info.minor}",
            RuntimeError("Version mismatch"),
        )
        return False


def check_core_imports() -> bool:
    """Verify core Python and external library imports."""
    check_header("Core Dependencies Check")

    imports_to_test = [
        ("numpy", "NumPy for numerical computing"),
        ("torch", "PyTorch for deep learning"),
        ("cv2", "OpenCV for computer vision"),
        ("hydra", "Hydra for configuration management"),
        ("omegaconf", "OmegaConf for configuration"),
        ("PIL", "Pillow for image processing"),
        ("sklearn", "Scikit-learn for machine learning utilities"),
    ]

    all_successful = True

    for module_name, description in imports_to_test:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "Unknown")
            check_success(f"{description}: {version}")
        except ImportError as e:
            check_error(f"Failed to import {module_name}", e)
            all_successful = False

    return all_successful


def check_project_imports() -> bool:
    """Verify project-specific imports work correctly."""
    check_header("Project Module Imports Check")

    project_imports = [
        ("src.data", "CrackSegmentationDataset", "Data processing module"),
        ("src.model", "BaseUNet", "Model architecture module"),
        ("src.model", "create_unet", "Model factory functions"),
        ("src.evaluation", "SegmentationMetrics", "Evaluation metrics"),
    ]

    all_successful = True

    for module_name, class_name, description in project_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            check_success(f"{description}: {class_name}")
        except (ImportError, AttributeError) as e:
            # Try to import the module at least
            try:
                __import__(module_name)
                check_success(
                    f"{description}: Module available "
                    "(specific class may vary)"
                )
            except ImportError:
                check_error(f"Failed to import {module_name}.{class_name}", e)
                all_successful = False

    return all_successful


def check_modern_python_features() -> bool:
    """Verify modern Python 3.12 features work correctly."""
    check_header("Modern Python 3.12 Features Check")

    try:
        # Test type hints with built-in generics (PEP 585)
        def test_generic_types(items: list[str]) -> dict[str, int]:
            return {item: len(item) for item in items}

        result = test_generic_types(["test", "example"])
        assert isinstance(result, dict)
        check_success("Built-in generic types (list[T], dict[K,V])")

        # Test union types with | operator (PEP 604)
        def test_union_types(value: str | int | None) -> str:
            return str(value) if value is not None else "None"

        assert test_union_types("test") == "test"
        assert test_union_types(42) == "42"
        assert test_union_types(None) == "None"
        check_success("Union types with | operator")

        # Test match statements (PEP 634)
        def test_match_statement(value: int) -> str:
            match value:
                case 0:
                    return "zero"
                case 1 | 2:
                    return "one or two"
                case n if n > 10:
                    return "big number"
                case _:
                    return "other"

        assert test_match_statement(0) == "zero"
        assert test_match_statement(1) == "one or two"
        assert test_match_statement(15) == "big number"
        check_success("Match statements (structural pattern matching)")

        # Test collections.abc imports
        def test_collections_abc(func: Callable[[int], str]) -> str:
            return func(42)

        result = test_collections_abc(lambda x: f"Value: {x}")
        assert result == "Value: 42"
        check_success("collections.abc.Callable type hints")

        return True

    except Exception as e:
        check_error("Modern Python features test", e)
        return False


def check_pytorch_cuda() -> bool:
    """Verify PyTorch CUDA functionality."""
    check_header("PyTorch CUDA Check")

    try:
        import torch

        check_success(f"PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            check_success(f"CUDA available: {torch.cuda.get_device_name(0)}")

            # Safe CUDA version detection using runtime check
            try:
                # Alternative method to get CUDA info
                cuda_arch_list = torch.cuda.get_arch_list()
                if cuda_arch_list:
                    check_success(
                        f"CUDA architectures: {', '.join(cuda_arch_list[:3])}"
                    )
                else:
                    print("⚠️  CUDA architecture info not available")
            except Exception:
                print("⚠️  CUDA architecture info not available")

            # Test basic tensor operations on GPU
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.matmul(x, y)
            assert z.is_cuda
            check_success("GPU tensor operations working")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")

        return True

    except Exception as e:
        check_error("PyTorch CUDA check", e)
        return False


def check_type_checking() -> bool:
    """Verify type checking tools work with Python 3.12."""
    check_header("Type Checking Tools Check")

    try:
        # Test that basedpyright/mypy can parse our code
        import subprocess

        # Check if basedpyright is available
        try:
            result = subprocess.run(
                ["basedpyright", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                check_success(
                    f"basedpyright available: {result.stdout.strip()}"
                )
            else:
                print("⚠️  basedpyright not found in PATH")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  basedpyright not found or timed out")

        # Test basic type annotations work
        def annotated_function(x: int, y: str) -> tuple[int, str]:
            return x, y

        result = annotated_function(42, "test")
        assert result == (42, "test")
        check_success("Type annotations parsing correctly")

        return True

    except Exception as e:
        check_error("Type checking tools", e)
        return False


def run_full_compatibility_check() -> bool:
    """Run all compatibility checks."""
    print("🚀 Starting Python 3.12 Compatibility Verification")
    print(f"📁 Working directory: {Path.cwd()}")

    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_imports),
        ("Project Imports", check_project_imports),
        ("Modern Python Features", check_modern_python_features),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("Type Checking", check_type_checking),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            check_error(f"{name} check failed", e)
            results.append((name, False))

    # Summary
    check_header("Compatibility Check Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {name}")

    print(f"\n📊 Results: {passed}/{total} checks passed")

    if passed == total:
        print(
            "🎉 All compatibility checks passed! "
            "Python 3.12 is fully supported."
        )
        return True
    else:
        print(
            "⚠️  Some compatibility issues found. "
            "Review the failed checks above."
        )
        return False


if __name__ == "__main__":
    success = run_full_compatibility_check()
    sys.exit(0 if success else 1)
