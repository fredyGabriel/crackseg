# Checkpoint Format Specification

This document defines the standardized checkpoint format for consistent model saving and loading
across the crack segmentation project.

## Overview

The standardized checkpoint format ensures that all training artifacts contain complete state
information required for model restoration, following consistent naming patterns and validation procedures.

## Checkpoint Structure

### Required Fields

All checkpoints must contain these fields for complete model restoration:

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | int | Training epoch number |
| `model_state_dict` | dict | PyTorch model state dictionary |
| `optimizer_state_dict` | dict | Optimizer state dictionary |
| `pytorch_version` | str | PyTorch version used for training |
| `timestamp` | str | ISO format timestamp of checkpoint creation |
| `config` | dict | Complete training configuration |

### Optional Fields

These fields enhance checkpoint utility but are not strictly required:

| Field | Type | Description |
|-------|------|-------------|
| `scheduler_state_dict` | dict | Learning rate scheduler state |
| `best_metric_value` | float | Best validation metric value |
| `metrics` | dict | Current epoch metrics |
| `python_version` | str | Python version information |
| `platform` | str | System platform information |
| `experiment_id` | str | Unique experiment identifier |
| `git_commit` | str | Git commit hash if available |
| `notes` | str | Additional notes or metadata |

## Filename Standards

### Pattern Format

Standardized filename pattern: `{base_name}_epoch_{epoch:03d}_{timestamp}.pth`

Examples:

- Regular checkpoint: `checkpoint_epoch_015_20240101_120000.pth`
- Best model: `model_best_epoch_010_20240101_120000.pth`
- Last checkpoint: `checkpoint_last_epoch_025_20240101_120000.pth`

### Naming Conventions

- Use `.pth` extension for all checkpoints
- Include zero-padded epoch number (3 digits)
- Include timestamp for uniqueness
- Use descriptive base names (`checkpoint`, `model`, etc.)

## Usage Examples

### Saving Standardized Checkpoint

```python
from crackseg.utils.checkpointing import save_checkpoint, CheckpointSaveConfig

config = CheckpointSaveConfig(
    checkpoint_dir="experiments/checkpoints",
    filename="checkpoint_epoch_010.pth",
    include_scheduler=True,
    include_python_info=True,
    validate_completeness=True,
)

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    config=config,
    scheduler=scheduler,
    best_metric_value=0.85,
    metrics={"train_loss": 0.1, "val_iou": 0.85},
    training_config=training_config_dict,
)
```

### Loading Checkpoint with Validation

```python
from crackseg.utils.checkpointing import load_checkpoint

# Load with strict validation
checkpoint_data = load_checkpoint(
    checkpoint_path="path/to/checkpoint.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    strict_validation=True,  # Enforce format compliance
)

print(f"Resumed from epoch {checkpoint_data['epoch']}")
print(f"Best metric: {checkpoint_data.get('best_metric_value', 'N/A')}")
```

### Verifying Checkpoint Integrity

```python
from crackseg.utils.checkpointing import verify_checkpoint_integrity

result = verify_checkpoint_integrity("path/to/checkpoint.pth")

if result["is_valid"]:
    print(f"✅ Valid checkpoint (epoch {result['epoch']})")
    print(f"   Size: {result['file_size_mb']:.1f} MB")
    print(f"   PyTorch: {result['pytorch_version']}")
else:
    print(f"❌ Invalid checkpoint")
    print(f"   Missing: {result['missing_required_fields']}")
    if result["error"]:
        print(f"   Error: {result['error']}")
```

### Legacy Checkpoint Adaptation

```python
from crackseg.utils.checkpointing import adapt_legacy_checkpoint, load_checkpoint

# Load old format checkpoint
legacy_data = load_checkpoint(
    checkpoint_path="old_checkpoint.pth",
    model=model,
    optimizer=optimizer,
    strict_validation=False,  # Don't enforce new format
)

# Adapt to standardized format
adapted_data = adapt_legacy_checkpoint(
    legacy_checkpoint=legacy_data,
    training_config=current_config,
)

# Now has all required metadata
assert "pytorch_version" in adapted_data
assert "timestamp" in adapted_data
```

## Configuration Options

### CheckpointSaveConfig Parameters

```python
@dataclass
class CheckpointSaveConfig:
    checkpoint_dir: str | Path              # Directory for checkpoint storage
    filename: str = "checkpoint.pt"         # Checkpoint filename
    additional_data: dict | None = None     # Extra data to include
    keep_last_n: int = 1                    # Number of recent checkpoints to keep
    include_scheduler: bool = True          # Include scheduler state
    include_python_info: bool = True        # Include Python/platform info
    validate_completeness: bool = True      # Validate against spec
```

### CheckpointSpec Customization

```python
from crackseg.utils.checkpointing import CheckpointSpec

# Custom specification for specialized use cases
custom_spec = CheckpointSpec()
custom_spec.required_fields.add("custom_field")
custom_spec.optional_fields.remove("git_commit")

# Use with validation
is_valid, missing = validate_checkpoint_completeness(checkpoint, custom_spec)
```

## Best Practices

### 1. Always Include Configuration

Store complete training configuration in checkpoints:

```python
training_config = {
    "model": {"architecture": "UNet", "num_classes": 2},
    "optimizer": {"type": "Adam", "lr": 0.001},
    "training": {"epochs": 100, "batch_size": 16},
    "data": {"dataset": "crack_dataset", "augmentations": [...]}
}

save_checkpoint(..., training_config=training_config)
```

### 2. Use Meaningful Filenames

```python
from crackseg.utils.checkpointing import create_standardized_filename

# Create descriptive, timestamped filenames
filename = create_standardized_filename(
    base_name="unet_crackseg",
    epoch=epoch,
    is_best=(current_iou > best_iou)
)
```

### 3. Validate Before Important Operations

```python
# Always verify checkpoints before deployment or sharing
result = verify_checkpoint_integrity(checkpoint_path)
if not result["is_valid"]:
    raise ValueError(f"Invalid checkpoint: {result['missing_required_fields']}")
```

### 4. Handle Legacy Checkpoints Gracefully

```python
try:
    # Try loading with strict validation first
    data = load_checkpoint(..., strict_validation=True)
except ValueError:
    # Fall back to legacy loading and adaptation
    data = load_and_adapt_legacy_checkpoint(...)
    logger.warning("Loaded legacy checkpoint, consider re-saving in new format")
```

## Migration Guide

### From Old Format

1. **Identify Legacy Checkpoints**: Use `verify_checkpoint_integrity()` to find incomplete checkpoints
2. **Load with Compatibility**: Use `strict_validation=False` for old checkpoints
3. **Adapt Format**: Use `adapt_legacy_checkpoint()` to add missing metadata
4. **Re-save Standardized**: Save adapted checkpoints in new format

### Batch Migration Script

```python
import glob
from pathlib import Path
from crackseg.utils.checkpointing import verify_checkpoint_integrity, load_and_adapt_legacy_checkpoint

def migrate_checkpoints(checkpoint_dir: str):
    """Migrate all checkpoints in directory to standardized format."""
    for ckpt_path in glob.glob(f"{checkpoint_dir}/*.pth"):
        result = verify_checkpoint_integrity(ckpt_path)

        if not result["is_valid"]:
            print(f"Migrating {ckpt_path}...")
            # Load and adapt, then re-save
            # (Implementation details depend on your specific use case)
```

## Troubleshooting

### Common Issues

1. **Missing Required Fields**: Use `validate_checkpoint_completeness()` to identify missing fields
2. **Incompatible PyTorch Versions**: Check `pytorch_version` field before loading
3. **Corrupted Checkpoints**: Use `verify_checkpoint_integrity()` for health checks
4. **Large Checkpoint Sizes**: Consider saving only essential state, exclude debug info

### Debugging Tools

```python
# Get detailed checkpoint information
result = verify_checkpoint_integrity(checkpoint_path)
print(f"Checkpoint info: {result}")

# Check what's actually in a checkpoint file
checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
print(f"Available fields: {list(checkpoint_data.keys())}")
```

## Version History

- **v1.0**: Initial standardized format implementation
- Added required fields: epoch, model_state_dict, optimizer_state_dict, pytorch_version, timestamp, config
- Added optional fields: scheduler_state_dict, best_metric_value, metrics, python_version, platform
- Implemented validation and verification tools
- Added legacy checkpoint adaptation support
