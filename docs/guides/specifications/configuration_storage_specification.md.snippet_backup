# Configuration Storage Specification

This document defines the standardized configuration storage system for consistent experiment
configuration management across the crack segmentation project.

## Overview

The standardized configuration storage system ensures that all training configurations are stored
with complete metadata, validation, and environment information for reproducible experiments.

## Key Features

### 1. Standardized Schema Validation

- **Required Fields**: Core fields that must be present in every configuration
- **Recommended Fields**: Optional but important fields for complete experiments
- **Environment Fields**: Automatically generated environment metadata

### 2. Automatic Environment Metadata

All configurations are enriched with:

- PyTorch version
- Python version
- Platform information
- CUDA availability and version
- Timestamp of configuration creation

### 3. Multiple Storage Formats

- **YAML**: Human-readable format (default)
- **JSON**: Machine-readable format for APIs

### 4. Configuration Comparison

- Compare configurations between experiments
- Identify differences in hyperparameters
- Ignore timestamp fields for meaningful comparisons

### 5. Legacy Migration Support

- Migrate old configuration formats to standardized format
- Add missing required fields with sensible defaults
- Preserve original configuration data

## Configuration Schema

### Required Fields

```yaml
experiment:
  name: "experiment_name"
model:
  _target_: "src.model.UNet"
training:
  epochs: 100
  optimizer:
    _target_: "torch.optim.Adam"
data:
  root_dir: "data/"
random_seed: 42
```

### Recommended Fields

```yaml
training:
  learning_rate: 0.001
  loss:
    _target_: "src.training.losses.BCEDiceLoss"
data:
  batch_size: 16
model:
  encoder:
    _target_: "src.model.encoders.ResNetEncoder"
  decoder:
    _target_: "src.model.decoders.UNetDecoder"
evaluation:
  metrics:
    - iou
    - f1_score
```

### Environment Metadata (Auto-generated)

```yaml
environment:
  pytorch_version: "2.5.1"
  python_version: "3.12.9"
  platform: "Windows-10-10.0.22631-SP0"
  cuda_available: true
  cuda_version: "12.4"
  cuda_device_count: 1
  cuda_device_name: "NVIDIA GeForce RTX 3080"
  timestamp: "2024-01-15T10:30:45.123456"

config_metadata:
  version: "1.0"
  schema_version: "1.0"
  created_at: "2024-01-15T10:30:45.123456"
  config_hash: "a1b2c3d4e5f6g7h8"
```

## Usage Examples

### Basic Configuration Storage

```python
from crackseg.utils.config.standardized_storage import StandardizedConfigStorage
from omegaconf import OmegaConf

# Initialize storage manager
storage = StandardizedConfigStorage("outputs/configurations")

# Load your configuration
config = OmegaConf.load("config/experiment.yaml")

# Save with validation and environment metadata
config_path = storage.save_configuration(
    config=config,
    experiment_id="exp_001",
    format_type="yaml"
)
```

### Loading and Comparing Configurations

```python
# Load configurations
config1 = storage.load_configuration("exp_001")
config2 = storage.load_configuration("exp_002")

# Compare configurations
comparison = storage.compare_configurations("exp_001", "exp_002")

if not comparison["are_identical"]:
    print(f"Found {comparison['total_differences']} differences:")
    for field, diff in comparison["differences"].items():
        print(f"  {field}: {diff['config1']} -> {diff['config2']}")
```

### Configuration Validation

```python
from crackseg.utils.config.standardized_storage import validate_configuration_completeness

# Validate configuration against schema
validation_result = validate_configuration_completeness(config)

if not validation_result["is_valid"]:
    print("Missing required fields:")
    for field in validation_result["missing_required"]:
        print(f"  - {field}")
```

### Legacy Configuration Migration

```python
from crackseg.utils.config.standardized_storage import migrate_legacy_configuration

# Migrate old configuration format
legacy_config = {
    "model": {"type": "unet"},
    "training": {"epochs": 50}
}

migrated_config = migrate_legacy_configuration(legacy_config)
# Now has required fields and environment metadata
```

## Directory Structure

The standardized storage creates the following structure:

```txt
outputs/configurations/
├── experiment_001/
│   ├── config.yaml
│   └── config_validation.json
├── experiment_002/
│   ├── config.yaml
│   └── config_validation.json
└── backups/
    ├── config_backup_exp_001_20240115_103045.yaml
    └── config_backup_exp_002_20240115_104512.yaml
```

## Integration with Existing Systems

### ExperimentManager Integration

```python

from crackseg.utils.config.standardized_storage import StandardizedConfigStorage

# Use with existing experiment manager
experiment_manager = ExperimentManager("outputs", "my_experiment")
storage = StandardizedConfigStorage(experiment_manager.config_dir)

# Save configuration with experiment context
storage.save_configuration(config, experiment_manager.experiment_id)
```

### Training Pipeline Integration

```python
# In training scripts
def main(config):
    # Initialize standardized storage
    storage = StandardizedConfigStorage("outputs/configurations")

    # Save configuration at start of training
    experiment_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage.save_configuration(config, experiment_id)

    # Continue with training...
```

## Validation Rules

### Required Field Validation

- All required fields must be present
- Fields cannot be None or empty
- Nested fields are validated using dot notation

### Schema Compliance

- Configuration must follow the defined schema structure
- Unknown fields are allowed but logged as warnings
- Type validation for critical fields

### Environment Consistency

- Environment metadata is automatically validated
- CUDA information is verified against actual hardware
- Version compatibility checks

## Best Practices

### 1. Consistent Naming

- Use descriptive experiment IDs
- Include date/time in experiment names
- Use semantic versioning for configuration versions

### 2. Configuration Backup

- Create backups before major changes
- Use version control for configuration templates
- Document configuration changes in commit messages

### 3. Validation Strategy

- Always validate configurations before training
- Use strict validation for production experiments
- Review validation reports for incomplete configurations

### 4. Comparison Workflow

- Compare configurations before reproducing experiments
- Document significant configuration differences
- Use configuration hashes for quick equality checks

## Error Handling

### Common Issues and Solutions

#### Missing Required Fields

```python
# Error: Configuration validation failed
# Solution: Add missing required fields
config.experiment = {"name": "my_experiment"}
config.random_seed = 42
```

#### Invalid Environment Metadata

```python
# Error: UnsupportedValueType for TorchVersion
# Solution: Already handled automatically by converting to string
```

#### File Permission Issues

```python
# Error: Permission denied when saving
# Solution: Check directory permissions or use different base directory
storage = StandardizedConfigStorage("/tmp/configurations")
```

## Migration Guide

### From Legacy Configurations

1. **Identify Legacy Format**: Check for old configuration structure
2. **Use Migration Tool**: Apply `migrate_legacy_configuration()`
3. **Validate Result**: Ensure all required fields are present
4. **Update Scripts**: Modify training scripts to use new format

### From Manual Configuration Storage

1. **Replace Manual Saving**: Remove custom `save_config()` calls
2. **Use StandardizedConfigStorage**: Initialize storage manager
3. **Update Loading Logic**: Use standardized loading methods
4. **Add Validation**: Include configuration validation in workflow

## Performance Considerations

- Configuration validation adds ~10ms overhead
- Environment metadata generation adds ~5ms overhead
- YAML format is slower than JSON but more readable
- Use JSON format for high-frequency configuration operations

## Security Considerations

- Configuration files may contain sensitive information
- Use appropriate file permissions (600 or 644)
- Avoid storing credentials in configuration files
- Consider encryption for sensitive configuration data

## Future Enhancements

- Configuration templates and inheritance
- Automatic hyperparameter optimization integration
- Configuration diff visualization tools
- Integration with experiment tracking systems
- Configuration schema evolution and migration tools
