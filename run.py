#!/usr/bin/env python
"""
Main project runner for pavement crack segmentation training and
evaluation. This script serves as the primary entry point for the
crack segmentation project, providing a robust wrapper around the main
training pipeline with proper environment configuration and error
handling. Key Features: - Automatic PYTHONPATH configuration for
project modules - Comprehensive error handling and logging - Seamless
integration with Hydra configuration system - Development-friendly
debugging and error reporting The script ensures that: 1. Project root
directory is added to PYTHONPATH for module import s 2. All
dependencies are properly loaded before execution 3. Detailed error
messages are provided for troubleshooting 4. Logging is configured for
transparent execution tracking Usage Examples: Basic training with
default configuration: ```bash python run.py ``` Training with custom
parameters: ```bash python run.py training.epochs=100
data.batch_size=8 ``` GPU training with specific device: ```bash
python run.py training.device=cuda:1 ``` Resume from checkpoint:
```bash python run.py training.checkpoints.resume_from_checkpoint=\
path/to/checkpoint.pth ``` Override configuration file: ```bash python
run.py --config-name=custom_config ``` Environment Requirements: -
Python 3.12+ with PyTorch and dependencies installed - CUDA support
recommended for GPU training - Sufficient disk space for datasets and
checkpoints Configuration: This script uses Hydra for configuration
management. The main config is located at 'configs/config.yaml' with
overrides via command line. Common configuration sections: - model:
Architecture and model parameters - data: Dataset paths and data
loading settings - training: Training hyperparameters and optimization
- evaluation: Metrics and evaluation configuration Error Handling: The
script provides detailed error messages for common issues: - Import
errors: Missing dependencies or incorrect installation - Configuration
errors: Invalid Hydra configurations - Runtime errors: Training
pipeline failures with full stack traces Note: This wrapper script is
the recommended way to run the project as it ensures proper
environment setup. Direct execution of src/main.py may result in
import errors if PYTHONPATH is not configured correctly. See Also: -
src/main.py: Core training pipeline implementation - configs/:
Configuration files and examples - docs/guides/WORKFLOW_TRAINING.md:
Detailed training workflow guide
"""

import logging
import sys
from pathlib import Path

from hydra import main as hydra_main

# Configure basic logging with detailed format for development
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("crackseg.runner")

# Ensure the project root directory and src directory are in PYTHONPATH for
# module imports
current_dir = str(Path(__file__).resolve().parent)
src_dir = str(Path(__file__).resolve().parent / "src")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"Added to PYTHONPATH: {current_dir}")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    logger.info(f"Added to PYTHONPATH: {src_dir}")


@hydra_main(version_base=None, config_path="configs", config_name="base")
def run_main(cfg) -> None:
    """
    Execute the main training function from src/main.py with error
    handling. This function serves as a robust wrapper around the main
    training pipeline, providing comprehensive error handling and
    informative logging for different types of failures that may occur
    during execution. The function handles several categories of errors: -
    Import errors: Missing dependencies, incorrect installations -
    Configuration errors: Invalid Hydra configurations - Runtime errors:
    Training pipeline failures, data loading issues - System errors:
    Resource constraints, permission issues Error Handling Strategy: 1.
    Import the main function with detailed error reporting 2. Execute
    training with comprehensive exception catching 3. Provide specific
    guidance for different error types 4. Exit gracefully with appropriate
    error codes Examples: Direct function call (not recommended):
    ```python from run import run_main run_main() # Better to use command
    line interface ``` Recommended usage via command line: ```bash python
    run.py # Executes this function automatically ``` Raises: SystemExit:
    With code 1 for any unrecoverable errors Note: This function is
    designed to be called from __main__ block and handles all errors
    internally, providing clean exit codes for process monitoring and
    automation scripts.
    """
    try:
        logger.info("Importing main function from src.main module...")
        from src.main import main

        logger.info("Starting main() execution with Hydra configuration...")

        # Handle nested configuration structure - SOLUTION DEFINITIVA
        # This resolves the root cause of the configuration nesting issue
        if hasattr(cfg, "experiments") and hasattr(
            cfg.experiments, "swinv2_hybrid"
        ):
            # Configuration is nested under experiments/swinv2_hybrid/
            # Move nested configuration to root level for compatibility
            logger.info(
                "Detected nested configuration, restructuring to root level..."
            )

            # Extract nested configuration and move to root level
            nested_config = cfg.experiments.swinv2_hybrid

            # Create new configuration with root-level structure
            from omegaconf import OmegaConf

            root_config = OmegaConf.create(
                {
                    "model": nested_config.model,
                    "data": nested_config.data,
                    "training": nested_config.training,
                    "evaluation": nested_config.evaluation,
                    "experiment": nested_config.experiment,
                    "save_config": nested_config.save_config,
                    # Preserve other root-level settings from base config
                    "project_name": cfg.get(
                        "project_name", "crack-segmentation"
                    ),
                    "output_dir": cfg.get("output_dir", "artifacts/"),
                    "data_dir": cfg.get("data_dir", "data/"),
                    "random_seed": cfg.get("random_seed", 42),
                    "seed": cfg.get("seed", 42),
                    "log_level": cfg.get("log_level", "INFO"),
                    "log_to_file": cfg.get("log_to_file", True),
                    "require_cuda": cfg.get("require_cuda", False),
                    "device": cfg.get("device", "auto"),
                    "memory": cfg.get("memory", {}),
                    "hardware": cfg.get("hardware", {}),
                    "timestamp_parsing": cfg.get("timestamp_parsing", {}),
                    "thresholds": cfg.get("thresholds", {}),
                    "visualization": cfg.get("visualization", {}),
                    "hydra": cfg.get("hydra", {}),
                }
            )

            logger.info("Configuration restructured successfully")
            main(root_config)
        else:
            # Configuration is already at root level
            main(cfg)
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
