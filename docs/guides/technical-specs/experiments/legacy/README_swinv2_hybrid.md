# SwinV2 Hybrid Architecture Experiment

## Overview

This experiment implements a state-of-the-art hybrid architecture for crack segmentation that
combines the strengths of three powerful components:

1. **SwinV2 Transformer Encoder**: Hierarchical feature extraction with shifted window attention
2. **ASPP Bottleneck**: Multi-scale context modeling for comprehensive crack understanding
3. **CNN Decoder**: Precise localization with skip connections for thin crack structures
4. **Focal Dice Loss**: Optimized for extreme class imbalance (<5% positive pixels)

## Architecture Details

### SwinV2CnnAsppUNet

The hybrid architecture follows this data flow:

```bash
Input Image (256x256x3)
    ↓
SwinV2 Encoder (swinv2_tiny_window16_256)
    ↓ (skip connections)
ASPP Bottleneck (rates: [1, 6, 12, 18])
    ↓
CNN Decoder + CBAM Attention
    ↓
Output (256x256x1) - Binary segmentation
```

**Key Features:**

- **SwinV2 Tiny**: Efficient transformer with 28M parameters
- **ASPP Bottleneck**: Captures multi-scale context (1-18 pixel receptive fields)
- **CBAM Attention**: Channel and spatial attention in decoder
- **Skip Connections**: U-Net style for fine detail preservation

### Focal Dice Loss

Optimized loss function for crack segmentation challenges:

```python
FocalDiceLoss = 0.6 * FocalLoss + 0.4 * DiceLoss

FocalLoss: Handles class imbalance (<5% positive pixels)
- alpha=0.25: Weight for positive class
- gamma=2.0: Focus on hard examples

DiceLoss: Optimizes segmentation metrics directly
- smooth=1.0: Numerical stability
- sigmoid=True: Apply sigmoid before computation
```

## Hardware Optimization

### RTX 3070 Ti (8GB VRAM) Settings

- **Batch Size**: 4 (effective 16 with gradient accumulation)
- **Mixed Precision**: Enabled (AMP) for memory efficiency
- **Gradient Accumulation**: 4 steps
- **Memory Usage**: ~6-7GB VRAM during training

### Performance Expectations

- **Training Time**: ~4-6 hours for 100 epochs
- **Memory Peak**: ~7GB VRAM
- **Expected IoU**: >0.85 on crack segmentation tasks
- **Expected F1**: >0.90

## Quick Start

### 1. Validate Configuration

```bash
# Dry run to validate configuration
python scripts/experiments/run_swinv2_hybrid_experiment.py --dry-run
```

### 2. Run Experiment

```bash
# Basic execution
python scripts/experiments/run_swinv2_hybrid_experiment.py

# With configuration overrides
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.epochs=50 \
    --config-override training.learning_rate=0.00005
```

### 3. Monitor Training

```bash
# TensorBoard logging (if enabled)
tensorboard --logdir artifacts/outputs/

# Check experiment logs
tail -f swinv2_hybrid_experiment.log
```

## Configuration Options

### Model Architecture

```yaml
model:
  encoder_cfg:
    model_name: "swinv2_tiny_window16_256"  # SwinV2 variant
    pretrained: true                        # ImageNet weights
    img_size: 256                          # Input size
  bottleneck_cfg:
    out_channels: 256                      # Bottleneck channels
    atrous_rates: [1, 6, 12, 18]          # Dilation rates
  decoder_cfg:
    use_cbam: true                         # Attention mechanism
    upsample_mode: "bilinear"              # Upsampling method
```

### Training Parameters

```yaml
training:
  loss:
    focal_weight: 0.6                      # Focal loss weight
    dice_weight: 0.4                       # Dice loss weight
    focal_alpha: 0.25                      # Class balance parameter
    focal_gamma: 2.0                       # Hard example focusing
  learning_rate: 0.0001                    # Learning rate
  batch_size: 4                            # Batch size
  gradient_accumulation_steps: 4           # Effective batch size = 16
  use_amp: true                            # Mixed precision
```

### Data Augmentation

```yaml
data:
  transform:
    augmentations:
      - A.RandomRotate90(p=0.5)           # Rotation
      - A.Flip(p=0.5)                     # Horizontal flip
      - A.ShiftScaleRotate(...)           # Geometric transforms
      - A.ElasticTransform(...)           # Elastic deformation
      - A.RandomBrightnessContrast(...)   # Photometric
      - A.CLAHE(p=0.3)                    # Contrast enhancement
```

## Expected Results

### Performance Metrics

Based on similar architectures and the crack segmentation domain:

| Metric | Expected Range | Target |
|--------|---------------|---------|
| IoU | 0.80 - 0.90 | >0.85 |
| F1 Score | 0.85 - 0.95 | >0.90 |
| Dice | 0.85 - 0.95 | >0.90 |
| Precision | 0.80 - 0.95 | >0.85 |
| Recall | 0.80 - 0.95 | >0.85 |

### Training Curves

Expected training behavior:

1. **Loss Convergence**: Focal Dice Loss should decrease steadily
2. **Validation Metrics**: IoU and F1 should improve over epochs
3. **Overfitting Check**: Monitor validation vs training metrics
4. **Early Stopping**: Should trigger around epoch 60-80

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.batch_size=2

# Increase gradient accumulation
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.gradient_accumulation_steps=8
```

#### 2. Slow Training

```bash
# Disable mixed precision (if causing issues)
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.use_amp=false

# Reduce image size
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override model.encoder_cfg.img_size=224
```

#### 3. Poor Convergence

```bash
# Adjust learning rate
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.learning_rate=0.00005

# Modify loss weights
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.loss.focal_weight=0.7 \
    --config-override training.loss.dice_weight=0.3
```

### Memory Monitoring

```python
# Monitor GPU memory during training
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"GPU Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
```

## Advanced Configuration

### Custom Loss Weights

```bash
# Experiment with different loss combinations
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override training.loss.focal_weight=0.8 \
    --config-override training.loss.dice_weight=0.2
```

### Different SwinV2 Variants

```bash
# Use larger model (requires more VRAM)
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override model.encoder_cfg.model_name=swinv2_small_window16_256

# Use smaller model (faster training)
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override model.encoder_cfg.model_name=swinv2_tiny_window8_256
```

### ASPP Configuration

```bash
# Adjust dilation rates for different crack scales
python scripts/experiments/run_swinv2_hybrid_experiment.py \
    --config-override model.bottleneck_cfg.atrous_rates=[1,3,6,9]
```

## Output Structure

After training, the experiment creates:

```bash
artifacts/outputs/YYYYMMDD-HHMMSS-swinv2_hybrid_focal_dice/
├── checkpoints/           # Model checkpoints
│   ├── best_model.pth    # Best validation model
│   └── latest_model.pth  # Latest model
├── config.yaml           # Complete configuration
├── metrics.json          # Training metrics
├── predictions/          # Validation predictions
├── logs/                 # Training logs
└── tensorboard/          # TensorBoard logs
```

## Comparison with Baselines

This hybrid architecture is expected to outperform:

| Architecture | Expected IoU | Expected F1 | Training Time |
|--------------|-------------|-------------|---------------|
| U-Net (ResNet50) | 0.75-0.80 | 0.80-0.85 | 2-3 hours |
| DeepLabV3+ | 0.80-0.85 | 0.85-0.90 | 3-4 hours |
| **SwinV2 Hybrid** | **0.85-0.90** | **0.90-0.95** | **4-6 hours** |

## References

- **Swin Transformer V2**: [Paper](https://arxiv.org/abs/2111.09883)
- **DeepLab v3+**: [Paper](https://arxiv.org/abs/1802.02611)
- **Focal Loss**: [Paper](https://arxiv.org/abs/1708.02002)
- **CBAM**: [Paper](https://arxiv.org/abs/1807.06521)

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the experiment logs in `swinv2_hybrid_experiment.log`
3. Verify GPU memory usage and system requirements
4. Ensure all dependencies are installed correctly

## Next Steps

After successful training:

1. **Evaluate on test set**: Use the trained model for inference
2. **Hyperparameter tuning**: Experiment with different configurations
3. **Model comparison**: Compare with other architectures
4. **Deployment**: Prepare model for production use
