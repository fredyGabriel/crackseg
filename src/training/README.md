# Training Module

This directory contains all core logic and helpers for model training, validation, and evaluation orchestration.

## Purpose
- Centralizes the training workflow, including epoch/batch processing, configuration validation, optimizer/scheduler factories, loss and metric definitions.
- Promotes modularity and separation of concerns for maintainable and extensible training code.

## File Overview
- `trainer.py`: Main Trainer class. Orchestrates the full training and validation loop, checkpointing, early stopping, and logging. Entry point for training logic.
- `batch_processing.py`: Stateless helpers for batch-level training and validation steps.
- `config_validation.py`: Functions to validate training configuration (Hydra/OmegaConf compatible).
- `factory.py`: Factory functions to instantiate optimizers and learning rate schedulers from config.
- `losses.py`: Loss function definitions for training (e.g., BCE, Dice, custom losses).
- `metrics.py`: Metric function definitions for evaluation (e.g., IoU, F1, accuracy).
- `__init__.py`: Module initialization.

## Conventions
- All configuration is loaded via Hydra/OmegaConf YAML files.
- No hardcoded hyperparameters: use config files for all training options.
- All new loss/metric functions should be stateless and registered in their respective modules.
- Use helpers/utilities from `src/utils/` for checkpointing, AMP, logging, and device management.

## Extending
- To add a new loss or metric: implement the function in `losses.py` or `metrics.py` and register it in the module's dictionary.
- To add new training logic: prefer helpers in `batch_processing.py` or new utility modules, keeping `trainer.py` focused on orchestration.

## Related
- See the main project README for high-level usage and configuration patterns.
- See `src/utils/` for shared utilities (logging, checkpointing, AMP, etc). 