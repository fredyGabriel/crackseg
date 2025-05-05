# Bottleneck Components

This directory contains configuration files for various bottleneck components that can be used in UNet-like architectures.

## Available Bottlenecks

- **default_bottleneck.yaml**: Simple convolutional bottleneck with a single convolution block
- **aspp_bottleneck.yaml**: Atrous Spatial Pyramid Pooling (ASPP) module that captures multi-scale context through parallel dilated convolutions
- **convlstm_bottleneck.yaml**: ConvLSTM-based bottleneck that can capture temporal dependencies
- **mock_bottleneck.yaml**: Mock implementation for testing

## Usage

Bottleneck components can be selected when composing model configurations:

```yaml
defaults:
  - _self_
  - bottleneck: aspp_bottleneck  # Select the desired bottleneck
```

## Key Parameters

Common parameters across bottleneck components:

| Parameter | Description |
|-----------|-------------|
| in_channels | Input channel count (should match encoder output) |
| output_channels | Output channel count (should match decoder input) |

## Custom Bottleneck Parameters

### ASPP Bottleneck
- `dilation_rates`: List of dilation rates for parallel convolutions
- `dropout_rate`: Dropout probability after feature fusion
- `output_stride`: Controls how dilations are scaled

### ConvLSTM Bottleneck
- `hidden_channels`: Number of hidden channels in ConvLSTM
- `kernel_size`: Kernel size for ConvLSTM convolutions
- `num_layers`: Number of ConvLSTM layers 