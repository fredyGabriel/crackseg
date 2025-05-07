# Project Source Code

This directory contains the main source code for the pavement crack segmentation project. The structure is modular and designed for maintainability, scalability, and reproducibility.

## Final Status

- All source code is organized by module and functionality.
- The structure supports modularity, extension, and experiment tracking.
- All modules are documented and versioned with the codebase.

## Directory Structure

- `data/`: Data handling and processing modules
  - `dataset.py`: Custom dataset implementations
  - `transforms.py`: Data transformations and augmentation
  - `factory.py`: Dataset factory
  - `memory.py`: Efficient memory utilities
  - `sampler.py`: Custom samplers
  - `splitting.py`: Data splitting functions
  - `distributed.py`: Distributed training support

- `model/`: Neural network architecture implementations
  - `unet.py`: U-Net architecture
  - `base.py`: Model base classes
  - `factory.py`: Model factory
  - `config.py`: Model-specific configurations
  - `encoder/`, `decoder/`, `bottleneck/`: Model components

- `training/`: Training and evaluation logic
  - `trainer.py`: Main training class
  - `losses.py`: Custom loss functions
  - `metrics.py`: Evaluation metrics
  - `factory.py`: Training component factory

- `utils/`: General utilities
  - `checkpointing.py`: Checkpoint management
  - `config_*.py`: Configuration utilities
  - `device.py`: Device management (CPU/GPU)
  - `early_stopping.py`: Early stopping implementation
  - `factory.py`: Generic factory
  - `logger_setup.py`: Logging configuration
  - `paths.py`: Path management
  - `seeds.py`: Random seed control

## Main Entry Point

- `main.py`: Main entry point that orchestrates training and evaluation

## Key Features

- Modular and extensible architecture
- Hydra-based configuration
- Full experiment support
- Detailed logging
- Efficient resource management
- Distributed training compatibility

## Usage

The project uses Hydra for configuration management. To run training:

```bash
python main.py
```

To modify configuration, use the files in the `configs/` directory or override parameters from the command line:

```bash
python main.py data.batch_size=32 training.epochs=100
``` 