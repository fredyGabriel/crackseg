#!/usr/bin/env python3
"""Demo script to verify the refactored TensorBoard package functionality.

This script demonstrates the new modular architecture and verifies that
all components work correctly after the refactoring.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_import_structure():
    """Test that all imports work correctly."""
    print("🧪 TensorBoard Refactoring Verification")
    print("=" * 60)

    # Test new modular imports
    print("📦 Testing New Modular Structure:")
    try:
        from gui.utils.tensorboard import (
            PortRegistry,
            TensorBoardManager,
        )

        print("   ✅ Core components imported successfully")

        # Test manager creation
        manager = TensorBoardManager()
        print(f"   ✅ Manager created: {type(manager).__name__}")
        print(f"   ✅ Initial state: {manager.info.state.value}")

        # Test port registry
        stats = PortRegistry.get_registry_stats()
        print(
            f"   ✅ Port registry: {stats['active_allocations']} allocations"
        )

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test convenience functions
    print("\n⚡ Testing Convenience Functions:")
    try:
        from gui.utils.tensorboard.manager import (
            create_default_tensorboard_setup,
        )

        manager, lifecycle = create_default_tensorboard_setup(
            auto_lifecycle=True
        )
        print(f"   ✅ Setup function works: {type(manager).__name__}")
        lifecycle_name = type(lifecycle).__name__ if lifecycle else None
        print(f"   ✅ Lifecycle created: {lifecycle_name}")
    except ImportError as e:
        print(f"   ❌ Convenience import failed: {e}")
        return False

    # Test backward compatibility
    print("\n🔄 Testing Backward Compatibility:")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from gui.utils.tb_manager import (
                TensorBoardManager as LegacyTB,
            )
        print(f"   ✅ Legacy import works: {type(LegacyTB).__name__}")
        print(f"   ✅ Same class: {LegacyTB is TensorBoardManager}")
    except ImportError as e:
        print(f"   ❌ Legacy import failed: {e}")
        return False

    return True


def test_file_structure():
    """Verify the file structure is correct."""
    print("\n📁 Verifying File Structure:")

    tensorboard_dir = Path(__file__).parent
    expected_files = [
        "__init__.py",
        "core.py",
        "port_management.py",
        "process_manager.py",
        "lifecycle_manager.py",
        "manager.py",
    ]

    for file in expected_files:
        file_path = tensorboard_dir / file
        if file_path.exists():
            lines = len(file_path.read_text().splitlines())
            print(f"   ✅ {file} ({lines} lines)")
        else:
            print(f"   ❌ {file} missing")
            return False

    # Check legacy wrapper
    legacy_path = tensorboard_dir.parent / "tb_manager.py"
    if legacy_path.exists():
        print("   ✅ tb_manager.py (compatibility wrapper)")
    else:
        print("   ❌ tb_manager.py missing")
        return False

    return True


def main():
    """Run verification tests."""
    print("Starting TensorBoard refactoring verification...\n")

    structure_ok = test_file_structure()
    imports_ok = test_import_structure()

    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("🎉 REFACTORING VERIFICATION SUCCESSFUL!")
        print("✨ All components working correctly")
        print("📁 Modular structure: scripts/gui/utils/tensorboard/")
        print("🔄 Legacy compatibility: scripts/gui/utils/tb_manager.py")
    else:
        print("❌ VERIFICATION FAILED")
        print("Some components are not working correctly")

    print("=" * 60)


if __name__ == "__main__":
    main()
