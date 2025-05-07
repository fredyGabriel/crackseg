# configs/model/

This directory contains YAML configuration files for segmentation model parameters and architectures.

## Structure

- `architectures/` — Complete architecture configurations (U-Net, Swin, hybrid, etc.)
- `bottleneck/` — Bottleneck module configurations
- `encoder/` — Encoder module configurations
- `decoder/` — Decoder module configurations
- `default.yaml` — Global default model configuration

## Notes

- Files named `mock_*.yaml` and `default_*.yaml` are for tests, examples, or as templates.
- If you add a new architecture, place it in `architectures/`.
- If you add a new component, place it in the corresponding subfolder.
- Keep this README updated to facilitate navigation and maintenance. 