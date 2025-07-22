# End-to-End (E2E) Testing Package

## Overview

This package provides comprehensive end-to-end testing utilities for the CrackSeg training pipeline.
It validates the complete workflow from data loading to model evaluation, ensuring all components
work together correctly.

## Structure

```bash
e2e/
├── test_pipeline_e2e.py    # Main E2E test script
├── modules/                # Supporting modules
│   ├── checkpointing.py    # Checkpoint management
│   ├── config.py          # Configuration generation
│   ├── dataclasses.py     # Data structures
│   ├── data.py            # Data utilities
│   ├── evaluation.py      # Model evaluation
│   ├── setup.py           # Initialization
│   ├── training.py        # Training components
│   └── utils.py           # General utilities
└── README.md              # This file
```

## Usage

### Running the E2E Test

```python
from scripts.experiments.e2e import run_e2e_test

# Run complete E2E test
experiment_dir, results = run_e2e_test()
```

### Using Individual Modules

```python
from scripts.experiments.e2e.modules import config, training, evaluation

# Create test configuration
cfg = config.create_mini_config()

# Initialize training components
model, loss_fn, optimizer = training._initialize_training_components(cfg, device)
```

## Test Components

### 1. Configuration (`config.py`)

- Generates minimal test configurations
- Validates configuration structure
- Provides synthetic dataset settings

### 2. Training (`training.py`)

- Model initialization
- Training loop execution
- Validation procedures
- Learning rate scheduling

### 3. Evaluation (`evaluation.py`)

- Model evaluation on test set
- Metrics calculation
- Sample prediction generation

### 4. Checkpointing (`checkpointing.py`)

- Checkpoint saving and loading
- Model state validation
- Results finalization

### 5. Data (`data.py`)

- Synthetic dataset generation
- DataLoader creation
- Data validation utilities

### 6. Setup (`setup.py`)

- Environment initialization
- Import management
- Logging configuration

### 7. Utils (`utils.py`)

- Configuration saving
- Visualization utilities
- Experiment directory management

## Test Workflow

1. **Setup**: Initialize environment, logging, and directories
2. **Configuration**: Generate minimal test configuration
3. **Data Loading**: Create synthetic dataset and dataloaders
4. **Training**: Execute training loop with validation
5. **Checkpointing**: Save and load model checkpoints
6. **Evaluation**: Evaluate model on test set
7. **Results**: Generate final report and visualizations

## Quality Standards

All modules follow project quality standards:

- ✅ Type annotations (Python 3.12+)
- ✅ Docstrings (Google Style)
- ✅ Line length limits (79 characters)
- ✅ Import organization
- ✅ Error handling

## Dependencies

- PyTorch
- OmegaConf
- Matplotlib
- YAML
- CrackSeg core modules

## Notes

- Uses synthetic data for fast testing
- Minimal configuration for quick validation
- Comprehensive logging and error reporting
- Modular design for easy maintenance
