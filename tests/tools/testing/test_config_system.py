#!/usr/bin/env python3
"""
Test script for the unified configuration loading system.

Quick verification that all new modules work together correctly.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    from gui.utils.config import (
        extract_config_value,
        generate_error_report,
        parse_nested_config,
        validate_crackseg_schema,
    )

    print("✅ All imports successful")

    # Test basic YAML parsing
    yaml_content = """
model:
  architecture: unet
  encoder: resnet50
  decoder: unet
  num_classes: 1

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam

data:
  data_root: "./data"
  image_size: [512, 512]
  batch_size: 16
"""

    print("\n🔧 Testing YAML parsing...")
    config, errors = parse_nested_config(yaml_content)

    if errors:
        print(f"❌ Parsing errors: {errors}")
    else:
        print("✅ YAML parsing successful")

        # Test value extraction
        learning_rate = extract_config_value(config, "training.learning_rate")
        print(f"✅ Extracted learning_rate: {learning_rate}")

    print("\n🔍 Testing schema validation...")
    # Validate the configuration
    is_valid, validation_errors, warnings = validate_crackseg_schema(config)
    assert is_valid
    assert not validation_errors
    assert not warnings

    # Generate a report (even if empty)
    error_report = generate_error_report(validation_errors, warnings)
    print(
        "Report generated with "
        f"{error_report['summary']['total_issues']} total issues"
    )

    # Test nested value extraction
    optimizer = extract_config_value(config, "training.optimizer.name")
    print(f"✅ Extracted optimizer: {optimizer}")

    print("\n🎯 **IMPLEMENTATION STATUS: FUNCTIONAL** ✅")
    print("All core components working correctly.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    sys.exit(1)
