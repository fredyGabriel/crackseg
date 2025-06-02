# Data Configurations

This directory contains configuration files for data-related components:

- `default.yaml`: General data configuration (paths, splits, batch sizes, etc.)
- `augmentations.yaml`: Data augmentation and transformation configuration
- `dataloader/default.yaml`: DataLoader configuration (batch size, sampler, memory optimizations, etc.)

## Batch Size and Num Workers

- The parameters `batch_size` and `num_workers` are defined only in `default.yaml`.
- All modules (including the DataLoader) must reference these values using Hydra interpolation.
- To change these parameters, edit `default.yaml`.
