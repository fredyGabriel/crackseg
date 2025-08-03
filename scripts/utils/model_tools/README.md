# Model Tools

This directory contains utilities for working with ML models.

## Scripts

### `model_summary.py`

Displays a summary of the U-Net model architecture.

- Loads a U-Net model from configuration
- Displays detailed summary including layer structure
- Shows parameter counts and memory usage estimates

### `unet_diagram.py`

Generates diagrams for U-Net architecture.

- Creates visual representations of model structure
- Helps understand model architecture
- Supports documentation and presentations

### `example_override.py`

Provides examples of model configuration overrides.

- Demonstrates Hydra configuration patterns
- Shows how to customize model parameters
- Serves as reference for model configuration

## Usage

```bash
# Generate model summary
python scripts/utils/model_tools/model_summary.py

# Generate U-Net diagram
python scripts/utils/model_tools/unet_diagram.py

# Run example override
python scripts/utils/model_tools/example_override.py
```

## Purpose

These utilities help with:

- Model understanding and analysis
- Architecture visualization
- Configuration examples
- Model documentation
