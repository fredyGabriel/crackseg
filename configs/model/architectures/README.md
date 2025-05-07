# Model Architectures

This directory contains complete model architecture configurations for the segmentation project. These YAML files define fully assembled models with their encoder, bottleneck, and decoder components.

## Directory Structure

All complete model architectures were moved here from the parent `model/` directory as part of a reorganization effort to improve code organization and maintenance. 

## Relationship to Component Configs

Each architecture file references component configurations that remain in their respective subdirectories:
- `model/encoder/` - Encoder configurations 
- `model/bottleneck/` - Bottleneck configurations
- `model/decoder/` - Decoder configurations

## Available Architectures

- `unet_cnn.yaml` - Standard CNN-based U-Net
- `unet_mock.yaml` - Mock implementation for testing
- `unet_aspp.yaml` - U-Net with ASPP bottleneck
- `unet_swin.yaml` - U-Net with Swin Transformer encoder
- `unet_swin_base.yaml` - U-Net with Swin-Base encoder
- `unet_swin_transfer.yaml` - U-Net with transfer learning from Swin
- `cnn_convlstm_unet.yaml` - ConvLSTM variant of U-Net
- `swinv2_hybrid.yaml` - Hybrid architecture with Swin Transformer v2

## Notes for Testing

When running tests that load these configurations, make sure to account for the new location. Some test files may need to be updated to use proper paths like `model/architectures/unet_cnn` instead of `model/unet_cnn`. 