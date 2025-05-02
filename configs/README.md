# Project Configurations

This directory contains all project configurations using Hydra. The structure is organized hierarchically to allow modular and flexible configuration.

## Directory Structure

- `base.yaml`: Base configuration that defines common parameters and default values
- `config.yaml`: Main configuration that composes different configuration groups

### Configuration Groups

- `data/`: Data-related configurations
  - `dataloader/`: Dataloader configuration
    - Batch size
    - Number of workers
    - Shuffling policies
  - `transform/`: Transformation configuration
    - Data augmentation
    - Normalization
    - Preprocessing

- `model/`: Architecture configurations
  - `encoder/`: Encoder configurations
  - `decoder/`: Decoder configurations
  - `bottleneck/`: Bottleneck configurations
  - Architecture hyperparameters
  - Weight initialization

- `training/`: Training configurations
  - `logging/`: Logger configuration
  - `loss/`: Loss function configuration
  - `lr_scheduler/`: Learning rate scheduler configuration
  - `metric/`: Metrics configuration
  - Training hyperparameters
  - Optimization policies

- `evaluation/`: Evaluation configurations
  - Evaluation metrics
  - Decision thresholds
  - Visualization parameters

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

## Configuration Structure

Example of hierarchical structure:

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
4. Keep experiment configurations reproducible
5. Version configurations along with code 