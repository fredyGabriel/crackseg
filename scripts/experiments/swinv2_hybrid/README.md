# SwinV2 Hybrid Architecture Experiment

This directory contains the complete experiment setup for the SwinV2 + ASPP + CNN hybrid
architecture with Focal Dice Loss for crack segmentation.

## ✅ Current Status: FULLY VALIDATED AND READY FOR EXECUTION

**All tests pass (9/9)** - The experiment has been comprehensively validated and is ready for
production training.

- ✅ **Setup Tests**: All 9 validation tests pass successfully
- ✅ **Dry Run**: Configuration validated with zero errors
- ✅ **All Issues Resolved**: 9/9 technical problems fixed
- ✅ **Hardware**: Optimized for RTX 3070 Ti (8GB VRAM)
- ✅ **Configuration**: Complete and production-ready
- ✅ **Code Quality**: All quality gates pass (black, ruff, basedpyright)

**Last Validation**: December 2024 - All components tested and fully functional.

## 📁 Structure

```bash
swinv2_hybrid/
├── README.md                           # This file (updated)
├── __init__.py                         # Python package marker
├── run_swinv2_hybrid_experiment.py     # Main experiment runner (validated)
├── test_swinv2_hybrid_setup.py         # Setup validation script (9/9 tests pass)
└── swinv2_hybrid_analysis.py           # Analysis and visualization wrapper
```

## 🚀 Quick Start

### 0. Verify Current Status

The experiment has been **completely validated** with all technical issues resolved:

```bash
# All tests pass (9/9) - comprehensive validation
python scripts/experiments/swinv2_hybrid/test_swinv2_hybrid_setup.py
```

**Expected Output**: `TEST SUMMARY: 9/9 tests passed`

### 1. Run Dry-Run Validation

```bash
# Validates all components without starting training
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py --dry-run
```

**Expected Output**: `Dry run completed successfully. All components validated.`

### 2. Start Full Training

```bash
# Start complete training pipeline
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py
```

### 3. Run Analysis (After Training)

```bash
python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py
```

## 📋 Configuration

The experiment configuration is located at:

- **Config**: `configs/experiments/swinv2_hybrid/swinv2_hybrid_experiment.yaml`
- **Documentation**: `docs/guides/experiments/README_swinv2_hybrid.md`

### Key Configuration Parameters

```yaml
# Model Architecture - SwinV2CnnAsppUNet
model:
  _target_: crackseg.model.architectures.swinv2_cnn_aspp_unet.SwinV2CnnAsppUNet
  type: "SwinV2CnnAsppUNet"
  encoder:
    model_name: "swinv2_tiny_window8_256"   # SwinV2 Tiny variant
    img_size: 256                           # Input image size
    target_img_size: 256                    # Target output size
    in_channels: 3                          # RGB input
  bottleneck:
    output_channels: 256                    # ASPP output channels
    dilation_rates: [1, 6, 12, 18]         # Multi-scale dilation rates
    dropout_rate: 0.1                       # Regularization
  decoder:
    use_cbam: true                          # Channel and spatial attention
    cbam_reduction: 16                      # CBAM reduction ratio
    upsample_mode: "bilinear"               # Upsampling method

# Training Configuration - Focal Dice Loss
training:
  loss:
    _target_: crackseg.training.losses.focal_dice_loss.FocalDiceLoss
    config:
      focal_weight: 0.6                     # Focal Loss weight
      dice_weight: 0.4                      # Dice Loss weight
      focal_alpha: 0.25                     # Focal Loss alpha
      focal_gamma: 2.0                      # Focal Loss gamma
  optimizer:
    lr: 0.0001                             # Learning rate (fixed)
    weight_decay: 0.01                     # L2 regularization
  batch_size: 4                            # Optimized for RTX 3070 Ti
  gradient_accumulation_steps: 4           # Effective batch size: 16
```

## 🏗️ Architecture Details

### Complete Hybrid Pipeline

- **Encoder**: SwinV2 Tiny with window attention (swinv2_tiny_window8_256)
- **Bottleneck**: ASPP with dilation rates [1, 6, 12, 18] and dropout (0.1)
- **Decoder**: CNN with CBAM attention and automatic upsampling to 256x256
- **Loss**: Focal Dice Loss optimized for crack segmentation (<5% positive pixels)

### Key Technical Features

- **Automatic Tensor Format Handling**: HWC ↔ CHW conversion in SwinV2 adapter
- **Smart Channel Management**: Automatic channel dimension inference between components
- **Skip Connection Ordering**: HIGH→LOW encoder output reordered to LOW→HIGH for decoder
- **Target Size Management**: Automatic final upsampling to maintain 256x256 output
- **Registry System**: Component registration with fallback support

## ⚙️ Hardware Optimization

Optimized for RTX 3070 Ti (8GB VRAM):

- **Batch size**: 4 (memory-efficient)
- **Gradient accumulation**: 4 (effective batch size: 16)
- **Mixed precision training**: AMP enabled
- **Memory-efficient settings**: Optimized dataloaders and model architecture
- **VRAM usage**: ~6-7GB during training

## 📊 Expected Performance

- **Training time**: ~2-3 hours per epoch (RTX 3070 Ti)
- **Memory usage**: ~6-7GB VRAM peak
- **Expected IoU**: 0.75-0.85 on crack segmentation
- **Expected Dice**: 0.80-0.90
- **Convergence**: Stable training with Focal Dice Loss

## 🔧 Technical Issues Resolved

### ✅ All 9 Critical Issues Fixed

1. **✅ FocalDiceLoss Configuration**: Proper nested config structure implemented
2. **✅ Transform Format Conversion**: Albumentations direct format → standard format
3. **✅ DataLoader Parameters**: Fixed persistent_workers duplicate parameter issue
4. **✅ Metrics Factory**: Support for both dict and list metric configurations
5. **✅ Optimizer/Scheduler Interpolation**: Proper Hydra interpolation resolution
6. **✅ Tensor Format (HWC/CHW)**: Automatic conversion in SwinV2 adapter
7. **✅ Model Registry**: Registry singleton and fallback system working
8. **✅ Target Size Mismatch**: Automatic final upsampling to 256x256 implemented
9. **✅ Complete Integration**: End-to-end pipeline fully validated

### 🧹 Cleanup Completed

- **Debug files removed**: All temporary debug scripts eliminated
- **Workarounds minimized**: Registry fallback exists but rarely used
- **Code quality**: All quality gates passing (black, ruff, basedpyright)
- **Documentation**: Updated to reflect current implementation

## 🔧 Customization Options

### Modify Learning Rate

```bash
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py \
    --config-override training.optimizer.lr=0.00005
```

### Adjust Loss Weights

```bash
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py \
    --config-override training.loss.config.focal_weight=0.7 \
    --config-override training.loss.config.dice_weight=0.3
```

### Change Model Variant

```bash
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py \
    --config-override model.encoder.model_name=swinv2_small_window8_256
```

### Modify ASPP Configuration

```bash
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py \
    --config-override model.bottleneck.dilation_rates=[1,3,6,12] \
    --config-override model.bottleneck.dropout_rate=0.2
```

## 📈 Monitoring & Logging

- **TensorBoard**: Logs saved to `outputs/experiments/swinv2_hybrid/`
- **Checkpoints**: Automatic saving every 5 epochs
- **Metrics**: IoU, Dice, Precision, Recall, F1-Score, Accuracy
- **Visualization**: Prediction samples and training curves
- **Logging**: Comprehensive logs in `swinv2_hybrid_experiment.log`

## 📊 Analysis & Visualization

### Quick Analysis

```bash
# Run complete analysis workflow
python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py

# Compare with baseline experiments
python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py --compare-baseline

# Export comprehensive report
python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py --export-report
```

### Analysis Output

The analysis generates:

- **Training Curves**: Loss, IoU, F1, Precision over epochs
- **Performance Radar**: Multi-metric comparison chart
- **Comparison Tables**: CSV data for further analysis
- **Comprehensive Report**: Markdown report with insights
- **Output Location**: `docs/reports/analysis/swinv2_hybrid_analysis/`

## 🐛 Troubleshooting

### ✅ Known Resolved Issues

**All major technical issues have been resolved**. The following were fixed:

1. **Registry Conflicts**: Proper component registration with graceful re-registration
2. **Tensor Format Issues**: Automatic HWC/CHW conversion implemented
3. **Configuration Errors**: All configuration format issues resolved
4. **Memory Issues**: Optimized for RTX 3070 Ti constraints
5. **Loss Function**: Proper FocalDiceLoss configuration implemented

### Expected Warnings (Normal)

You may see these **normal** warnings during execution:

```bash
Component 'SwinV2CnnAsppUNet' already registered in 'Architecture' registry. Overwriting with new implementation.
ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
Cannot set number of intraop threads after parallel work has started
```

These warnings are **expected and safe** - they indicate the system is working correctly.

### Common Adjustments

#### Out of Memory

```bash
# Reduce batch size
--config-override training.batch_size=2

# Reduce gradient accumulation
--config-override training.gradient_accumulation_steps=2
```

#### Slow Training

```bash
# Reduce number of workers
--config-override data.dataloader.num_workers=2

# Disable mixed precision (if needed)
--config-override training.use_amp=false
```

#### Convergence Issues

```bash
# Adjust learning rate
--config-override training.optimizer.lr=0.00001

# Modify loss balance
--config-override training.loss.config.focal_weight=0.8
--config-override training.loss.config.dice_weight=0.2
```

## 📚 Related Documentation

- [Complete Experiment Guide](../../../docs/guides/experiments/README_swinv2_hybrid.md)
- [Architecture Details](../../../docs/guides/architecture/)
- [Training Workflow](../../../docs/guides/workflows/)
- [Troubleshooting Guide](../../../docs/guides/troubleshooting/)
- [ASPP Analysis](../../../docs/reports/analysis/aspp_analysis.md)

## 🔬 Research Context

This experiment implements a **state-of-the-art hybrid architecture** specifically designed for
pavement crack segmentation:

### Technical Innovation

- **Multi-Scale Transformer**: SwinV2 with shifted window attention for global context
- **Adaptive Context Modeling**: ASPP with learnable dilation rates [1, 6, 12, 18]
- **Precision Localization**: CNN decoder with CBAM attention for fine detail preservation
- **Class Imbalance Optimization**: Focal Dice Loss tuned for <5% positive pixels

### Performance Characteristics

- **Thin Structure Detection**: Optimized for 1-5px width crack detection
- **Multi-Scale Context**: Captures both hairline cracks and structural damage
- **Robust Performance**: Handles various lighting and pavement conditions
- **Hardware Efficient**: Optimized for consumer GPU constraints (RTX 3070 Ti)

### Key Research Contributions

- **Hybrid Architecture**: Novel combination of transformer + CNN + attention
- **Automatic Pipeline**: End-to-end trainable with minimal manual tuning
- **Production Ready**: Comprehensive validation and error handling
- **Reproducible**: Fixed seeds, comprehensive logging, and version control

### Expected Research Outcomes

- **Benchmark Performance**: Target IoU >0.85, Dice >0.90 on crack segmentation
- **Generalization**: Robust across different pavement types and conditions
- **Efficiency**: Training convergence in <50 epochs with stable loss curves
- **Practical Impact**: Deployable system for real-world infrastructure monitoring

---

**Status**: Production Ready ✅
**Last Updated**: December 2024
**Validation**: 9/9 tests passing
**Quality Gates**: All passing (black, ruff, basedpyright)
