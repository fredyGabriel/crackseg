# configs/model/

This directory contains YAML configuration files for segmentation model parameters and architectures.

## Final Status

- All model and architecture configuration files are organized by component and purpose.
- The structure supports modularity, reuse, and easy extension for new architectures or components.
- All files are documented and versioned with the codebase.

## Structure

- `architectures/` — Complete architecture configurations (U-Net, Swin, hybrid, etc.)
- `bottleneck/` — Bottleneck module configurations
- `encoder/` — Encoder module configurations
- `decoder/` — Decoder module configurations
- `default.yaml` — Global default model configuration

## Best Practices

- Files named `mock_*.yaml` and `default_*.yaml` are for tests, examples, or as templates.
- When adding a new architecture, place it in `architectures/` and document its purpose.
- When adding a new component, place it in the corresponding subfolder and update this README.
- Keep configuration files minimal, focused, and well-documented to facilitate navigation and
  maintenance.
