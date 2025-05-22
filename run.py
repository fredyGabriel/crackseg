#!/usr/bin/env python
"""
Script to run main.py ensuring proper environment configuration.

This script:
1. Configures PYTHONPATH to include the project root directory
2. Executes the main function from src/main.py with Hydra

Usage:
    python run.py [hydra_options]
"""

import logging
import sys
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("crackseg.runner")

# Ensure the project root directory is in PYTHONPATH
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"Added to PYTHONPATH: {current_dir}")


def run_main():
    """Execute the main function from src/main.py."""
    try:
        logger.info("Importing main function...")
        from src.main import main

        logger.info("Starting main() execution...")
        main()
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Verify that all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_main()
