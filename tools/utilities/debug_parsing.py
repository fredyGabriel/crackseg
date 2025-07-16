#!/usr/bin/env python3
"""Debug script to test comment parsing."""

import tempfile
from pathlib import Path

from tests.docker.env_manager import EnvironmentManager

# Create a test file with comments
temp_file = Path(tempfile.mktemp())
content_with_comments = """
NODE_ENV=test
LOG_LEVEL=info                    # Log level (debug, info, warn, error)
DEBUG=false
TEST_BROWSER=firefox # Comment here too
"""
temp_file.write_text(content_with_comments)

print("Original content:")
print(content_with_comments)
print("=====")

# Test parsing
manager = EnvironmentManager()
try:
    print(f"Reading file: {temp_file}")
    print(f"File exists: {temp_file.exists()}")
    size = temp_file.stat().st_size if temp_file.exists() else "N/A"
    print(f"File size: {size}")
    if temp_file.exists():
        print(f"Content:\n{temp_file.read_text()}")

    env_vars = manager.load_env_file(temp_file)
    print(f"Returned dict length: {len(env_vars)}")
    print("Parsed variables:")
    for k, v in env_vars.items():
        print(f"  {k}={v!r}")

    # Test specific LOG_LEVEL
    if "LOG_LEVEL" in env_vars:
        log_val = env_vars["LOG_LEVEL"]
        print("\nLOG_LEVEL analysis:")
        print(f"  Raw value: {log_val!r}")
        print(f"  Length: {len(log_val)}")
        print(f"  Contains hash: {'#' in log_val}")
        print(f"  Stripped: {log_val.strip()!r}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    temp_file.unlink(missing_ok=True)
