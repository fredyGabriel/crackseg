#!/usr/bin/env python
"""
Main entry point for running the CrackSeg project from the command line.
"""

import os
import sys

# Ensure the package root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust if __main__.py is deeper
package_root = os.path.abspath(os.path.join(current_dir, ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from crackseg.main import main  # noqa: E402

if __name__ == "__main__":
    # This allows the package to be run as a script
    # Example: python -m src
    main()
