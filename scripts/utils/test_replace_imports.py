#!/usr/bin/env python3
"""
Test script for the import replacement functionality.

This script tests the ImportReplacer class with sample files to ensure
the replacement process works correctly before applying to actual documentation.

Author: CrackSeg Development Team
Date: 2025-01-27
"""

import os
import shutil
import tempfile

from replace_imports import ImportReplacer


def create_test_file(content: str, filename: str = "test.md") -> str:
    """Create a temporary test file with the given content."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def test_basic_replacement():
    """Test basic import replacement functionality."""
    print("Testing basic replacement...")

    # Create test content with import statements
    test_content = """# Test Documentation

This is a test file with import statements.

```python
from src.crackseg.utils.deployment.config import DeploymentConfig
from src.crackseg.utils.deployment.orchestration import DeploymentOrchestrator
from src.crackseg.evaluation.simple_prediction_analyzer import SimplePredictionAnalyzer
```

More content here.
"""

    # Create test file
    test_file = create_test_file(test_content)

    try:
        # Create replacer in dry-run mode
        replacer = ImportReplacer(dry_run=True, backup=False, verbose=True)

        # Override target files to use our test file
        replacer.target_files = [test_file]

        # Run the replacement
        success = replacer.run()

        # Check results
        assert success, "Replacement should succeed"
        assert replacer.stats["files_processed"] == 1, "Should process 1 file"
        assert replacer.stats["files_modified"] == 1, "Should modify 1 file"
        assert (
            replacer.stats["total_replacements"] == 3
        ), "Should make 3 replacements"

        print("✅ Basic replacement test passed")

    finally:
        # Cleanup
        shutil.rmtree(os.path.dirname(test_file))


def test_no_changes_needed():
    """Test when no replacements are needed."""
    print("Testing no changes scenario...")

    # Create test content without import statements
    test_content = """# Test Documentation

This is a test file without import statements.

```python
import torch
import numpy as np
```

More content here.
"""

    # Create test file
    test_file = create_test_file(test_content)

    try:
        # Create replacer in dry-run mode
        replacer = ImportReplacer(dry_run=True, backup=False, verbose=True)

        # Override target files to use our test file
        replacer.target_files = [test_file]

        # Run the replacement
        success = replacer.run()

        # Check results
        assert success, "Replacement should succeed"
        assert replacer.stats["files_processed"] == 1, "Should process 1 file"
        assert (
            replacer.stats["files_modified"] == 0
        ), "Should not modify any files"
        assert (
            replacer.stats["total_replacements"] == 0
        ), "Should make 0 replacements"

        print("✅ No changes test passed")

    finally:
        # Cleanup
        shutil.rmtree(os.path.dirname(test_file))


def test_backup_functionality():
    """Test backup file creation."""
    print("Testing backup functionality...")

    # Create test content
    test_content = """# Test Documentation

```python
from src.crackseg.utils.deployment.config import DeploymentConfig
```
"""

    # Create test file
    test_file = create_test_file(test_content)

    try:
        # Create replacer with backup enabled
        replacer = ImportReplacer(dry_run=False, backup=True, verbose=True)

        # Override target files to use our test file
        replacer.target_files = [test_file]

        # Run the replacement
        success = replacer.run()

        # Check results
        assert success, "Replacement should succeed"
        assert replacer.stats["files_modified"] == 1, "Should modify 1 file"

        # Check that backup file was created
        backup_file = f"{test_file}.backup"
        assert os.path.exists(backup_file), "Backup file should exist"

        # Verify backup content matches original
        with open(backup_file, encoding="utf-8") as f:
            backup_content = f.read()
        assert (
            backup_content == test_content
        ), "Backup should contain original content"

        print("✅ Backup functionality test passed")

    finally:
        # Cleanup
        shutil.rmtree(os.path.dirname(test_file))


def test_live_replacement():
    """Test actual file modification."""
    print("Testing live replacement...")

    # Create test content
    test_content = """# Test Documentation

```python
from src.crackseg.utils.deployment.config import DeploymentConfig
from src.crackseg.utils.deployment.orchestration import DeploymentOrchestrator
```
"""

    # Create test file
    test_file = create_test_file(test_content)

    try:
        # Create replacer in live mode
        replacer = ImportReplacer(dry_run=False, backup=False, verbose=True)

        # Override target files to use our test file
        replacer.target_files = [test_file]

        # Run the replacement
        success = replacer.run()

        # Check results
        assert success, "Replacement should succeed"
        assert replacer.stats["files_modified"] == 1, "Should modify 1 file"
        assert (
            replacer.stats["total_replacements"] == 2
        ), "Should make 2 replacements"

        # Verify file was actually modified
        with open(test_file, encoding="utf-8") as f:
            modified_content = f.read()

        # Check that imports were replaced
        assert (
            "from crackseg.utils.deployment.config import DeploymentConfig"
            in modified_content
        )
        assert (
            "from crackseg.utils.deployment.orchestration import DeploymentOrchestrator"
            in modified_content
        )
        assert (
            "from src.crackseg.utils.deployment.config import DeploymentConfig"
            not in modified_content
        )
        assert (
            "from src.crackseg.utils.deployment.orchestration import DeploymentOrchestrator"
            not in modified_content
        )

        print("✅ Live replacement test passed")

    finally:
        # Cleanup
        shutil.rmtree(os.path.dirname(test_file))


def test_error_handling():
    """Test error handling for non-existent files."""
    print("Testing error handling...")

    # Create replacer
    replacer = ImportReplacer(dry_run=True, backup=False, verbose=True)

    # Use non-existent file
    replacer.target_files = ["non_existent_file.md"]

    # Run the replacement
    success = replacer.run()

    # Check results
    assert not success, "Should fail due to non-existent file"
    assert replacer.stats["errors"] == 1, "Should have 1 error"

    print("✅ Error handling test passed")


def main():
    """Run all tests."""
    print("Running import replacement tests...\n")

    tests = [
        test_basic_replacement,
        test_no_changes_needed,
        test_backup_functionality,
        test_live_replacement,
        test_error_handling,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")

    print(f"\nTest Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
