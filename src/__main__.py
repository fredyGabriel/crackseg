#!/usr/bin/env python
"""
Main entry point for running the CrackSeg project from the command line.
"""

import os
import sys

# Ensure the project root directory is in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# After setting up sys.path, we can safely import
from src.main import main  # noqa: E402

if __name__ == "__main__":
    main()
