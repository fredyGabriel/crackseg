"""Simple test script for the mapping registry.

This script tests the basic functionality of the mapping registry without
depending on complex project imports.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

# Import the registry directly
sys.path.insert(0, str(project_root / "src" / "crackseg" / "utils"))
from mapping_registry import MappingRegistry  # noqa: E402


def test_basic_functionality() -> None:
    """Test basic registry functionality."""
    print("🧪 Testing Mapping Registry Basic Functionality")
    print("=" * 50)

    # Create a test registry
    registry = MappingRegistry()

    # Add some test mappings
    registry.add_mapping("old.path", "new.path", "import", "Test mapping")

    registry.add_mapping(
        "outputs/", "artifacts/", "artifact", "Test artifact mapping"
    )

    # Test getting mappings
    mapping = registry.get_mapping("old.path", "import")
    if mapping:
        print(f"✅ Found mapping: {mapping.old_path} -> {mapping.new_path}")
    else:
        print("❌ Mapping not found")

    # Test applying mappings
    test_content = "from old.path import something"
    result = registry.apply_mapping(test_content, "import")
    print(f"✅ Applied mapping: '{test_content}' -> '{result}'")

    # Test statistics
    stats = registry.get_statistics()
    print(f"✅ Registry statistics: {stats}")

    # Test validation
    errors = registry.validate_mappings()
    if errors:
        print(f"❌ Validation errors: {errors}")
    else:
        print("✅ No validation errors")

    print()


def test_file_processing() -> None:
    """Test processing files with mappings."""
    print("🧪 Testing File Processing")
    print("=" * 50)

    registry = MappingRegistry()

    # Add test mappings
    registry.add_mapping(
        "from src.crackseg", "from crackseg", "import", "Standardize imports"
    )

    registry.add_mapping(
        "outputs/", "artifacts/", "artifact", "Standardize artifact paths"
    )

    # Test content processing
    test_contents = [
        "from src.crackseg.model import UNet",
        'output_dir = "outputs/experiments"',
        "import src.crackseg.utils",
        'checkpoint_path = "outputs/checkpoints/model.pth"',
    ]

    for content in test_contents:
        result = registry.apply_mapping(content, "import")
        result = registry.apply_mapping(result, "artifact")
        print(f"'{content}' -> '{result}'")

    print()


def test_registry_persistence() -> None:
    """Test saving and loading the registry."""
    print("🧪 Testing Registry Persistence")
    print("=" * 50)

    # Create registry and add mappings
    registry = MappingRegistry()
    registry.add_mapping("old", "new", "import", "Test")
    registry.add_mapping("outputs", "artifacts", "artifact", "Test")

    # Save to temporary file
    test_file = Path("test_registry.json")
    registry.save_registry(test_file)
    print(f"✅ Saved registry to {test_file}")

    # Create new registry and load
    new_registry = MappingRegistry()
    new_registry.load_registry(test_file)
    print(f"✅ Loaded registry with {len(new_registry.mappings)} mappings")

    # Clean up
    test_file.unlink(missing_ok=True)
    print("✅ Cleaned up test file")

    print()


def main() -> None:
    """Run all tests."""
    print("🚀 Starting Mapping Registry Tests")
    print("=" * 60)
    print()

    try:
        test_basic_functionality()
        test_file_processing()
        test_registry_persistence()

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
