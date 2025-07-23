# Data Module

This directory contains all data handling and processing components for the pavement crack
segmentation project. The module provides a comprehensive framework for loading, transforming, and
managing crack segmentation datasets.

## Module Overview

The data module is designed with modularity and efficiency in mind, supporting various data loading
strategies, augmentation techniques, and memory optimization approaches.

### Core Components

- **`dataset.py`**: Main dataset implementation (`CrackSegmentationDataset`)
- **`dataloader.py`**: DataLoader configuration and optimization
- **`transforms.py`**: Data augmentation and preprocessing pipelines
- **`factory.py`**: Factory functions for creating datasets and dataloaders
- **`validation.py`**: Data validation and integrity checking
- **`splitting.py`**: Dataset splitting utilities (train/val/test)
- **`memory.py`**: Memory optimization and caching utilities
- **`sampler.py`**: Custom sampling strategies
- **`distributed.py`**: Distributed training support

## Key Features

### Dataset Management

- **Flexible Loading**: Support for various image and mask formats
- **Automatic Validation**: Image-mask correspondence checking
- **Memory Optimization**: In-memory caching and efficient loading
- **Split Management**: Configurable train/validation/test splits

### Data Augmentation

- **Training Augmentations**: HorizontalFlip, VerticalFlip, Rotate, ColorJitter
- **Normalization**: ImageNet statistics for transfer learning
- **Validation Transforms**: Consistent preprocessing for evaluation
- **Custom Pipelines**: Extensible transformation framework

### Performance Optimization

- **Multi-processing**: Configurable worker processes
- **Memory Management**: Adaptive batch sizing and caching
- **Distributed Support**: Multi-GPU training compatibility
- **Prefetching**: Optimized data loading pipelines

## Usage Examples

### Basic Dataset Creation

```python
from crackseg.data.dataset import CrackSegmentationDataset
from crackseg.data.transforms import get_transforms_from_config

# Create dataset with transforms
transforms = get_transforms_from_config('train')
dataset = CrackSegmentationDataset(
    data_dir='data/train',
    transforms=transforms
)

# Access data
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # (3, 512, 512)
print(f"Mask shape: {mask.shape}")    # (1, 512, 512)
```

### DataLoader Configuration

```python
from crackseg.data.dataloader import create_dataloader
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('configs/data/default.yaml')

# Create optimized dataloader
dataloader = create_dataloader(
    dataset=dataset,
    config=config.dataloader,
    split='train'
)

# Training loop
for batch_idx, (images, masks) in enumerate(dataloader):
    # Training logic here
    pass
```

### Factory Pattern Usage

```python
from crackseg.data.factory import create_datasets, create_dataloaders

# Create all datasets from configuration
datasets = create_datasets(config)
train_dataset = datasets['train']
val_dataset = datasets['val']

# Create all dataloaders
dataloaders = create_dataloaders(datasets, config)
train_loader = dataloaders['train']
val_loader = dataloaders['val']
```

### Custom Transforms

```python
from crackseg.data.transforms import create_transform_pipeline

# Create custom transform pipeline
custom_transforms = create_transform_pipeline([
    {'name': 'Resize', 'params': {'height': 256, 'width': 256}},
    {'name': 'HorizontalFlip', 'params': {'p': 0.5}},
    {'name': 'Normalize', 'params': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }},
    {'name': 'ToTensorV2', 'params': {}}
])
```

## Configuration Integration

The data module integrates seamlessly with Hydra configuration:

```yaml
# configs/data/default.yaml
data:
  data_root: data/
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  image_size: [512, 512]
  batch_size: 16
  num_workers: 4
  in_memory_cache: false

  dataloader:
    shuffle: true
    pin_memory: true
    prefetch_factor: 2
    drop_last: false
```

## Data Format Requirements

### Directory Structure

```txt
data/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── masks/      # Binary masks (.png)
├── val/
│   ├── images/     # Validation images
│   └── masks/      # Validation masks
└── test/
    ├── images/     # Test images
    └── masks/      # Test masks
```

### Image Specifications

- **Formats**: PNG, JPG, JPEG
- **Channels**: 3 (RGB) for images, 1 (grayscale) for masks
- **Mask Values**: 0 (background), 255 (crack)
- **Naming**: Corresponding images and masks must have matching filenames

### Data Validation

```python
from crackseg.data.validation import validate_data_config

# Validate dataset integrity
validation_results = validate_data_config('data/train')
if validation_results['valid']:
    print("Dataset validation passed")
else:
    print(f"Validation errors: {validation_results['errors']}")
```

## Memory Optimization

### Caching Strategies

```python
# Enable in-memory caching for small datasets
dataset = CrackSegmentationDataset(
    data_dir='data/train',
    in_memory_cache=True
)

# Use memory mapping for large datasets
dataset = CrackSegmentationDataset(
    data_dir='data/train',
    memory_map=True
)
```

### Batch Size Optimization

```python
from crackseg.data.memory import optimize_batch_size

# Automatically determine optimal batch size
optimal_batch_size = optimize_batch_size(
    model=model,
    dataset=dataset,
    device='cuda:0',
    max_memory_gb=8
)
```

## Distributed Training Support

```python
from crackseg.data.distributed import create_distributed_sampler

# Create distributed sampler for multi-GPU training
sampler = create_distributed_sampler(
    dataset=dataset,
    num_replicas=world_size,
    rank=rank
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=sampler
)
```

## Performance Monitoring

### Data Loading Profiling

```python


# Profile dataloader performance
stats = profile_dataloader(dataloader, num_batches=100)
print(f"Average batch time: {stats['avg_batch_time']:.3f}s")
print(f"Data loading efficiency: {stats['efficiency']:.2%}")
```

### Memory Usage Tracking

```python
from crackseg.data.memory import track_memory_usage

# Monitor memory usage during data loading
with track_memory_usage() as tracker:
    for batch in dataloader:
        # Process batch
        pass

print(f"Peak memory usage: {tracker.peak_memory_mb:.1f} MB")
```

## Testing and Validation

The data module includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end data pipeline testing
- **Performance Tests**: Memory and speed benchmarks
- **Validation Tests**: Data integrity and format checking

```bash
# Run data module tests
pytest tests/unit/data/ -v
pytest tests/integration/data/ -v
```

## Best Practices

1. **Data Validation**: Always validate datasets before training
2. **Memory Management**: Monitor memory usage and optimize batch sizes
3. **Reproducibility**: Set seeds for deterministic data loading
4. **Error Handling**: Implement robust error handling for missing files
5. **Performance**: Profile data loading to identify bottlenecks
6. **Configuration**: Use Hydra configs for all data parameters

## Troubleshooting

### Common Issues

**Slow Data Loading**:

- Increase `num_workers` (but not beyond CPU cores)
- Enable `pin_memory` for GPU training
- Use SSD storage for datasets
- Consider in-memory caching for small datasets

**Memory Issues**:

- Reduce `batch_size`
- Disable `pin_memory` if system RAM is limited
- Use memory mapping for large datasets
- Monitor memory usage with profiling tools

**Data Validation Errors**:

- Check image-mask filename correspondence
- Verify image and mask formats
- Ensure consistent directory structure
- Validate mask values (0 and 255 only)

## Related Documentation

- **Configuration Guide**: `configs/data/README.md`
- **Training Workflow**: `docs/guides/WORKFLOW_TRAINING.md`
- **Data Directory**: `data/README.md`
- **Project Structure**: `.cursor/rules/project-structure.mdc`
