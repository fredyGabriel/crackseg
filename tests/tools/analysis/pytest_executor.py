from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class PytestExecutor:
    """Executes pytest and captures its output."""

    def __init__(self, test_paths: list[str], timeout: int = 300):
        self.test_paths = test_paths
        self.timeout = timeout

    def run(self) -> str:
        """Executes pytest and returns the combined stdout and stderr."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *self.test_paths,
            "--tb=long",
            "-v",
            "--capture=no",
            "--maxfail=50",
            "--no-header",
            "--disable-warnings",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path.cwd(),
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            print(f"⏰ Test execution timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            print(f"❌ Error executing tests: {e}")
            raise
