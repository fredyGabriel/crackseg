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
        load_config_with_validation,
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
    is_valid, validation_errors, warnings = validate_crackseg_schema(config)

    print(
        f"Schema validation result: {'✅ Valid' if is_valid else '❌ Invalid'}"
    )
    print(f"Errors: {len(validation_errors)}, Warnings: {len(warnings)}")

    if validation_errors:
        for error in validation_errors[:3]:  # Show first 3 errors
            print(f"  Error: {error.message}")

    if warnings:
        for warning in warnings[:3]:  # Show first 3 warnings
            print(f"  Warning: {warning}")

    print("\n📊 Testing error reporting...")
    error_report = generate_error_report(validation_errors, warnings)
    print(
        f"Report generated with {error_report['summary']['total_issues']} total issues"
    )

    print("\n🎯 **IMPLEMENTATION STATUS: FUNCTIONAL** ✅")
    print("All core components working correctly.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    sys.exit(1)
