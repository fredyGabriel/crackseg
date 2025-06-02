# Data Configurations

This directory contains configuration files for data-related components in the pavement crack segmentation project.

## Configuration Files

### Core Configuration

- **`default.yaml`**: Main data configuration including:
  - Data paths and directory structure
  - Train/validation/test splits (70%/15%/15%)
  - Batch size and image dimensions
  - Memory and caching settings
  - Expected tensor dimensions and channels

### Data Processing

- **`transform/augmentations.yaml`**: Data augmentation and transformation configuration including:
  - Training augmentations (HorizontalFlip, VerticalFlip, Rotate, ColorJitter)
  - Validation/test transformations (Resize, Normalize)
  - Normalization parameters (ImageNet statistics)
  - ToTensorV2 conversion

### DataLoader Configuration

- **`dataloader/default.yaml`**: DataLoader configuration including:
  - Batch processing parameters
  - Worker processes and memory optimization
  - Distributed training settings
  - Sampling strategies
  - Memory management (FP16, adaptive batch size)

## Key Parameters

### Batch Size and Workers

- **`batch_size`**: Defined in `default.yaml` (default: 16)
- **`num_workers`**: Defined in `default.yaml` (default: 4)
- All modules reference these values using Hydra interpolation: `${data.batch_size}`

### Image Processing

- **Image size**: `[512, 512]` (configurable)
- **Input channels**: 3 (RGB)
- **Output channels**: 1 (binary segmentation)
- **Expected dimensions**: 4D tensors `(N, C, H, W)`

### Data Splits

- **Training**: 70% of dataset
- **Validation**: 15% of dataset
- **Test**: 15% of dataset

## Usage Examples

### Override batch size for 8GB VRAM

```bash
python run.py data.batch_size=4
```

### Change image resolution

```bash
python run.py data.image_size=[256,256]
```

### Modify data augmentation

```bash
python run.py data.transform=augmentations
```

### Enable in-memory caching

```bash
python run.py data.in_memory_cache=true
```

## Configuration Structure

```yaml
# Example structure from default.yaml
data:
  data_root: data/
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  image_size: [512, 512]
  batch_size: 16
  num_workers: 4
  # ... additional parameters
```

## Integration

These configurations integrate with:

- **DataLoader**: `src/data/dataloader.py`
- **Dataset**: `src/data/dataset.py`
- **Transforms**: `src/data/transforms.py`
- **Training Pipeline**: `src/training/trainer.py`

## Best Practices

1. **Centralized Parameters**: Always define `batch_size` and `num_workers` in `default.yaml`
2. **Hydra Interpolation**: Use `${data.parameter}` to reference values across configs
3. **Memory Optimization**: Adjust `batch_size` and `num_workers` based on available hardware
4. **Consistent Splits**: Maintain the same data splits across experiments for reproducibility

## Hardware Recommendations

- **RTX 3070 Ti (8GB)**: `batch_size=4`, `num_workers=4`
- **RTX 4090 (24GB)**: `batch_size=16`, `num_workers=8`
- **CPU Only**: `batch_size=2`, `num_workers=2`
