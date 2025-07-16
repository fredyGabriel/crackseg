"""
Test script for config_io module functionality.
"""

from pathlib import Path

from gui.utils.config_io import (
    ConfigError,
    get_config_metadata,
    load_config_file,
    scan_config_directories,
    validate_yaml_syntax,
)


def test_scan_directories():
    """Test scanning configuration directories."""
    print("Testing scan_config_directories()...")
    configs = scan_config_directories()

    print(f"\nFound {len(configs)} categories:")
    for category, files in configs.items():
        print(f"\n{category}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"  - {Path(file).name}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")


def test_load_config():
    """Test loading a configuration file."""
    print("\n\nTesting load_config_file()...")

    # Try to load the base config
    try:
        config = load_config_file("configs/base.yaml")
        print("Successfully loaded configs/base.yaml")
        print(f"Config keys: {list(config.keys())}")
    except ConfigError as e:
        print(f"Error loading config: {e}")


def test_yaml_validation():
    """Test YAML syntax validation."""
    print("\n\nTesting validate_yaml_syntax()...")

    # Valid YAML
    valid_yaml = """
model:
  type: unet
  encoder: resnet50

training:
  epochs: 100
  batch_size: 16
"""

    is_valid, error = validate_yaml_syntax(valid_yaml)
    print(f"Valid YAML test: {is_valid}")

    # Invalid YAML
    invalid_yaml = """
model:
  type: unet
  encoder resnet50  # Missing colon
"""

    is_valid, error = validate_yaml_syntax(invalid_yaml)
    print(f"Invalid YAML test: Valid={is_valid}, Error={error}")


def test_metadata():
    """Test getting file metadata."""
    print("\n\nTesting get_config_metadata()...")

    # Get metadata for base.yaml
    metadata = get_config_metadata("configs/base.yaml")
    print("Metadata for configs/base.yaml:")
    print(f"  - Size: {metadata.get('size_human', 'N/A')}")
    print(f"  - Modified: {metadata.get('modified', 'N/A')}")
    preview = metadata.get("preview", [])
    if isinstance(preview, list):
        preview_len = len(preview)
    else:
        preview_len = 0
    print(f"  - Preview lines: {preview_len}")


if __name__ == "__main__":
    print("Config I/O Module Test Suite")
    print("=" * 50)

    test_scan_directories()
    test_load_config()
    test_yaml_validation()
    test_metadata()

    print("\n\nAll tests completed!")
