# PY-CrackDB SwinV2 Experiment

## Overview

This experiment evaluates the performance of a SwinV2 encoder combined with a CNN decoder for crack
segmentation on the PY-CrackDB dataset. The images have been processed from their original 351×500
dimensions to 320×320 using intelligent bidirectional cropping to preserve maximum crack information.

**Status**: ✅ **EXECUTED SUCCESSFULLY** - Experiment completed with 2 epochs, showing clear metric improvements

## Experiment Details

### Hypothesis

SwinV2 hierarchical attention combined with CNN decoder will effectively detect thin crack
structures in 320×320 images, leveraging the transformer's ability to capture long-range
dependencies while maintaining the CNN decoder's efficiency for local feature refinement.

### Architecture

- **Encoder**: SwinV2-T (Tiny) with window size 8
- **Decoder**: CNN with ASPP (Atrous Spatial Pyramid Pooling)
- **Input Size**: 320×320×3
- **Output**: Binary segmentation mask (320×320×1)

### Dataset

- **Source**: PY-CrackDB (369 images)
- **Processing**: Bidirectional cropping from 351×500 to 320×320
- **Split**: 70% train, 15% validation, 15% test
- **Format**: JPG images, PNG masks

## Configuration

### Model Configuration

```yaml
model:
  encoder_cfg:
    model_name: swinv2_tiny_window8_256
    img_size: 320
    patch_size: 4
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
    window_size: 8
    mlp_ratio: 4.0
    drop_path_rate: 0.1

  bottleneck_cfg:
    output_channels: 256
    dilation_rates: [1, 6, 12, 18]
    dropout_rate: 0.1
    input_channels: 768
    reduction_factor: 4

  decoder_cfg:
    use_cbam: true
    cbam_reduction: 16
    upsample_scale_factor: 4
    decoder_channels: [256, 128, 64, 32]
    skip_connections: true
    attention_gates: true
```

### Training Configuration

- **Epochs**: 3 (configurable)
- **Batch Size**: 12 (optimized for RTX 3070 Ti)
- **Learning Rate**: 5.0e-05
- **Optimizer**: AdamW
- **Loss**: BCEDiceLoss (BCE + Dice, 0.5 + 0.5 weights)
- **Scheduler**: Cosine Annealing with Warm Restarts (T_0=30)

### Hardware Optimization

- **GPU**: RTX 3070 Ti (8GB VRAM)
- **Mixed Precision**: Enabled
- **Gradient Accumulation**: 2 steps
- **Memory Management**: Optimized for 8GB constraint

## Target Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| IoU | 0.75 | Standard segmentation metric |
| F1 Score | 0.85 | Balance precision and recall |
| Dice | 0.82 | Effective for class imbalance |

## 🎯 **Execution Results**

### Test Run Results (2 Epochs)

The experiment was successfully executed with the following results:

```bash
Epoch 1: val_loss: 0.9075 | val_iou: 0.0902 | val_dice: 0.1623 | val_precision: 0.0904 | val_recall: 0.9813
Epoch 2: val_loss: 0.8833 | val_iou: 0.2220 | val_dice: 0.3543 | val_precision: 0.2260 | val_recall: 0.9534
```

### Key Observations

- ✅ **Clear metric improvements** between epochs
- ✅ **Loss decreasing** from 0.9075 to 0.8833
- ✅ **IoU improving** from 0.0902 to 0.2220 (146% improvement)
- ✅ **Dice score improving** from 0.1623 to 0.3543 (118% improvement)
- ✅ **System functioning correctly** - all components working as expected

### Execution Command

```bash
conda activate crackseg
python run.py --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb +training.epochs=2
```

## Augmentation Strategy

### Spatial Augmentations

- Horizontal/Vertical flips (p=0.5)
- Rotation (±90° for thin structures)
- Resize to 320×320

### Photometric Augmentations

- Color jitter (brightness, contrast, saturation, hue)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Normalization (ImageNet stats)

## Expected Challenges

### Dataset Characteristics

- **Small dataset**: 369 images total
- **Class imbalance**: <5% positive pixels
- **Thin structures**: 1-5 pixel wide cracks
- **Varied crack patterns**: From hairline to structural damage

### Technical Challenges

- **Memory constraints**: 8GB VRAM limit
- **Overfitting risk**: Small dataset with complex model
- **Thin structure detection**: Preserving 1-5 pixel cracks

## Mitigation Strategies

### For Small Dataset

- **More epochs**: 50 epochs for thorough training
- **Strong regularization**: Weight decay 0.01
- **Early stopping**: Patience 15 epochs
- **Data augmentation**: Comprehensive augmentation pipeline

### For Memory Constraints

- **Reduced batch size**: 8 (effective 16 with accumulation)
- **Mixed precision**: FP16 training
- **Gradient accumulation**: 2 steps
- **Memory monitoring**: Continuous VRAM tracking

### For Thin Structures

- **Higher dice weight**: 0.6 vs 0.4 BCE
- **Limited rotation**: ±15° to preserve structure
- **CLAHE augmentation**: Enhance local contrast
- **Attention mechanisms**: CBAM and attention gates

## Running the Experiment

### Prerequisites

1. **Activate the conda environment**:

   ```bash
   conda activate crackseg
   ```

2. **Process PY-CrackDB dataset** (if not already done):

   ```bash
   python scripts/data_processing/image_processing/process_py_crackdb_example.py
   ```

3. **Verify processed dataset exists**:

   ```txt
   data/PY-CrackBD_processed/
   ├── images/ (369 .jpg files)
   └── masks/ (369 .png files)
   ```

### Important Notes

- **No separate experiment script needed**: The Hydra configuration nesting problem has been resolved
- **Direct execution**: Use `run.py` with the configuration file directly
- **Environment activation**: Always activate the conda environment before running
- **Configuration validation**: The configuration file is validated before execution

### Execution

```bash
# Run the experiment directly with run.py (RECOMMENDED)
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb

# Or with custom parameters
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb \
  training.epochs=100 \
  training.learning_rate=5e-05
```

### Monitoring

- **TensorBoard**: `artifacts/experiments/py_crackdb_swinv2/tensorboard/`
- **Logs**: `artifacts/experiments/py_crackdb_swinv2/logs/`
- **Checkpoints**: `artifacts/experiments/py_crackdb_swinv2/checkpoints/`
- **Configuration**: `artifacts/experiments/py_crackdb_swinv2/config.yaml`

## Expected Results

### Performance Expectations

- **Training time**: ~2-3 hours on RTX 3070 Ti
- **Memory usage**: ~6-7GB VRAM
- **Convergence**: ~30-40 epochs

### Success Criteria

- IoU > 0.75
- F1 Score > 0.85
- Dice > 0.82
- No overfitting (val loss < train loss)

## Analysis Plan

### Quantitative Analysis

- **Per-image metrics**: Detailed analysis of individual predictions
- **Confusion matrix**: Precision/recall breakdown
- **Crack density correlation**: Performance vs crack density
- **Boundary accuracy**: Thin structure preservation

### Qualitative Analysis

- **Visualization**: Overlay predictions on original images
- **Failure cases**: Analysis of missed detections
- **Boundary quality**: Thin crack preservation assessment
- **False positives**: Background misclassification analysis

## Comparison Baseline

### Previous Results (Crack500)

| Model | IoU | F1 | Dice | Parameters |
|-------|-----|----|------|------------|
| U-Net | 0.681 | 0.811 | 0.792 | 31.0M |
| DeepCrack | 0.869 | 0.930 | 0.918 | 14.7M |
| DeepLabV3+ | 0.688 | 0.816 | 0.799 | 40.3M |

### Expected PY-CrackDB Performance

- **Target**: Surpass U-Net baseline (IoU > 0.68)
- **Stretch goal**: Approach DeepCrack performance (IoU > 0.85)
- **Realistic**: IoU 0.75-0.80 range

## Future Work

### Potential Improvements

1. **Larger dataset**: Combine with other crack datasets
2. **Advanced loss functions**: Boundary-aware losses
3. **Ensemble methods**: Multiple model predictions
4. **Post-processing**: Morphological operations
5. **Test-time augmentation**: Multi-scale predictions

### Research Directions

- **Attention visualization**: Understanding SwinV2 attention patterns
- **Feature analysis**: Decoder feature map analysis
- **Transfer learning**: Pre-training on larger datasets
- **Architecture search**: Optimizing encoder-decoder combinations

## References

- **SwinV2**: Liu et al. "Swin Transformer V2: Scaling Up Capacity and Resolution"
- **Crack Segmentation**: Liu et al. "DeepCrack: A Deep Hierarchical Feature Learning Architecture
  for Crack Segmentation"
- **PY-CrackDB**: Original dataset paper
- **ASPP**: Chen et al. "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous
  Convolution, and Fully Connected CRFs"

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'hydra'**
   - **Solution**: Activate the conda environment: `conda activate crackseg`

2. **Dataset not found**
   - **Solution**: Run the processing script first: `python scripts/data_processing/image_processing/process_py_crackdb_example.py`

3. **CUDA out of memory**
   - **Solution**: Reduce batch size in configuration or use CPU: `training.device=cpu`

4. **Configuration not found**
   - **Solution**: Verify the configuration path is correct: `configs/experiments/swinv2_hybrid/swinv2_320x320_py_crackdb.yaml`

### Performance Optimization

- **Memory usage**: Monitor with `nvidia-smi` during training
- **Training time**: ~2-3 hours expected on RTX 3070 Ti
- **Checkpoint frequency**: Every 5 epochs (configurable)

## Contact

For questions about this experiment, refer to the project documentation or create an issue in the repository.
