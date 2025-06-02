#!/usr/bin/env python
"""
Main project runner for pavement crack segmentation training and evaluation.

This script serves as the primary entry point for the crack segmentation
project, providing a robust wrapper around the main training pipeline with
proper environment configuration and error handling.

Key Features:
- Automatic PYTHONPATH configuration for project modules
- Comprehensive error handling and logging
- Seamless integration with Hydra configuration system
- Development-friendly debugging and error reporting

The script ensures that:
1. Project root directory is added to PYTHONPATH for module imports
2. All dependencies are properly loaded before execution
3. Detailed error messages are provided for troubleshooting
4. Logging is configured for transparent execution tracking

Usage Examples:
    Basic training with default configuration:
        ```bash
        python run.py
        ```

    Training with custom parameters:
        ```bash
        python run.py training.epochs=100 data.batch_size=8
        ```

    GPU training with specific device:
        ```bash
        python run.py training.device=cuda:1
        ```

    Resume from checkpoint:
        ```bash
        python run.py training.checkpoints.resume_from_checkpoint=\
            path/to/checkpoint.pth
        ```

    Override configuration file:
        ```bash
        python run.py --config-name=custom_config
        ```

Environment Requirements:
    - Python 3.12+ with PyTorch and dependencies installed
    - CUDA support recommended for GPU training
    - Sufficient disk space for datasets and checkpoints

Configuration:
    This script uses Hydra for configuration management. The main config
    is located at 'configs/config.yaml' with overrides via command line.

    Common configuration sections:
    - model: Architecture and model parameters
    - data: Dataset paths and data loading settings
    - training: Training hyperparameters and optimization
    - evaluation: Metrics and evaluation configuration

Error Handling:
    The script provides detailed error messages for common issues:
    - Import errors: Missing dependencies or incorrect installation
    - Configuration errors: Invalid Hydra configurations
    - Runtime errors: Training pipeline failures with full stack traces

Note:
    This wrapper script is the recommended way to run the project as it
    ensures proper environment setup. Direct execution of src/main.py
    may result in import errors if PYTHONPATH is not configured correctly.

See Also:
    - src/main.py: Core training pipeline implementation
    - configs/: Configuration files and examples
    - docs/guides/WORKFLOW_TRAINING.md: Detailed training workflow guide
"""

import logging
import sys
from pathlib import Path

# Configure basic logging with detailed format for development
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("crackseg.runner")

# Ensure the project root directory is in PYTHONPATH for module imports
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"Added to PYTHONPATH: {current_dir}")


def run_main() -> None:
    """
    Execute the main training function from src/main.py with error handling.

    This function serves as a robust wrapper around the main training pipeline,
    providing comprehensive error handling and informative logging for
    different types of failures that may occur during execution.

    The function handles several categories of errors:
    - Import errors: Missing dependencies, incorrect installations
    - Configuration errors: Invalid Hydra configurations
    - Runtime errors: Training pipeline failures, data loading issues
    - System errors: Resource constraints, permission issues

    Error Handling Strategy:
    1. Import the main function with detailed error reporting
    2. Execute training with comprehensive exception catching
    3. Provide specific guidance for different error types
    4. Exit gracefully with appropriate error codes

    Examples:
        Direct function call (not recommended):
        ```python
        from run import run_main
        run_main()  # Better to use command line interface
        ```

        Recommended usage via command line:
        ```bash
        python run.py  # Executes this function automatically
        ```

    Raises:
        SystemExit: With code 1 for any unrecoverable errors

    Note:
        This function is designed to be called from __main__ block
        and handles all errors internally, providing clean exit codes
        for process monitoring and automation scripts.
    """
    try:
        logger.info("Importing main function from src.main...")
        from src.main import main

        logger.info("Starting main() execution with Hydra configuration...")
        main()
        logger.info("Training pipeline completed successfully!")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "This typically indicates missing dependencies or incorrect "
            "installation."
        )
        logger.error(
            "Please verify that all dependencies are installed correctly:"
        )
        logger.error("  1. Check environment.yml or requirements.txt")
        logger.error(
            "  2. Ensure PyTorch is installed with CUDA support if needed"
        )
        logger.error("  3. Verify all src/ modules are available")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
        logger.error("Full stack trace for debugging:")
        import traceback

        logger.error(traceback.format_exc())
        logger.error(
            "This error occurred in the main training pipeline. "
            "Check configuration files and data paths."
        )
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting crack segmentation training runner...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    run_main()
