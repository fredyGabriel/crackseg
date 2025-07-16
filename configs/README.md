# Project Configurations

This directory contains all configuration files for the project, managed using Hydra. The structure
is hierarchical and modular to ensure flexibility, clarity, and reproducibility for all experiments
and workflows.

## Final Status

- All configuration files are organized by module and purpose.
- The structure supports modular overrides and experiment tracking.
- All parameters are documented and versioned with the codebase.

## Directory Structure

- `base.yaml`: Base configuration with common parameters and default values
- `config.yaml`: Main configuration that composes different configuration groups

### Configuration Groups

- `data/`: Data-related configurations
  - `dataloader/`: Dataloader settings (batch size, workers, shuffling)
  - `transform/`: Data augmentation, normalization, preprocessing

- `model/`: Model and architecture configurations
  - `encoder/`, `decoder/`, `bottleneck/`: Component-specific settings
  - Architecture hyperparameters, weight initialization

- `training/`: Training configurations
  - `logging/`: Logger settings
  - `loss/`: Loss function settings
  - `lr_scheduler/`: Learning rate scheduler
  - `metric/`: Metrics for training/validation
  - Training hyperparameters, optimization policies

- `evaluation/`: Evaluation configurations
  - Metrics, thresholds, visualization parameters

## Using with Hydra

Configurations can be combined and overridden using Hydra syntax:

```bash
# Use a specific configuration
python main.py +experiment=unet_baseline

# Override specific parameters
python main.py training.optimizer.lr=0.001

# Combine multiple modifications
python main.py +experiment=unet_baseline training.batch_size=32 data.augmentation=strong
```

## Configuration Structure Example

```yaml
defaults:
  - _self_
  - data: default
  - model: unet
  - training: default
  - evaluation: default
```

## Best Practices

1. Keep configurations modular and reusable
2. Document all parameters and their effects
3. Use sensible default values
4. Ensure experiment configurations are reproducible
5. Version configurations along with code
6. Use Hydra's override and composition features for flexible experimentation
