# SwinV2 Hybrid Experiment - 360x360 Optimized

## Overview

This experiment trains the SwinV2 hybrid architecture (SwinV2 + ASPP + CNN decoder) on 360x360 pixel
images from the unified dataset, optimized for RTX 3070 Ti hardware with 8GB VRAM.

## Architecture

- **Encoder**: SwinV2-tiny with window size 8
- **Bottleneck**: ASPP module with dilation rates [1, 6, 12, 18]
- **Decoder**: CNN decoder with CBAM attention
- **Loss**: FocalDiceLoss (focal_weight=0.6, dice_weight=0.4)
- **Optimizer**: AdamW with cosine annealing scheduler

## Dataset

- **Source**: `data/unified/` (1,754 images, 360x360 pixels)
- **Split**: 70% train (1,228), 15% val (263), 15% test (263)
- **Format**: JPG images, PNG masks (binarized)

## Hardware Optimizations

### RTX 3070 Ti (8GB VRAM) Optimizations

- **Batch Size**: 2 (reduced from 4 for 360x360 images)
- **Gradient Accumulation**: 8 steps (effective batch size = 16)
- **Mixed Precision**: AMP enabled for memory efficiency
- **Gradient Checkpointing**: Enabled for memory optimization
- **Memory Monitoring**: Continuous monitoring and optimization

### Memory Management

```python
# Theoretical memory usage for 360x360 images
Model Parameters: ~28M (SwinV2-tiny)
Feature Maps: ~2.3GB (batch_size=2, 360x360)
Gradients: ~112MB
Optimizer States: ~112MB
Total Estimated: ~2.5GB + overhead
```

## Configuration Files

### Main Configuration

- **File**: `configs/experiments/swinv2_hybrid/swinv2_hybrid_360x360_experiment.yaml`
- **Purpose**: Complete experiment configuration optimized for 360x360

### Key Optimizations

```yaml
# Memory optimizations
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  use_amp: true

# Hardware settings
hardware:
  cudnn_benchmark: false
  cudnn_deterministic: true
  mixed_precision: true

# Memory management
memory:
  gradient_checkpointing: true
  empty_cache_freq: 100
  monitor_memory: true
```

## Execution

### Prerequisites

1. **Dataset**: Ensure `data/unified/` contains processed 360x360 images
2. **Environment**: Activate conda environment
3. **Dependencies**: All project dependencies installed

### Running the Experiment

```bash
# Navigate to project root
cd /path/to/crackseg

# Run the experiment
python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_360x360_experiment.py

# Monitor training
tail -f artifacts/experiments/*/training.log
```

### Expected Output

```txt
SwinV2 Hybrid Experiment - 360x360 Optimized
================================================================================
Configuration:
  Model: SwinV2CnnAsppUNet
  Image Size: [360, 360]
  Batch Size: 2
  Gradient Accumulation: 8
  Effective Batch Size: 16
  Mixed Precision: True
  Epochs: 100

Model parameters: 28,000,000 total, 28,000,000 trainable
Training samples: 1228
Validation samples: 263
```

## Monitoring and Analysis

### Real-time Monitoring

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Monitor training progress
tail -f artifacts/experiments/*/training.log

# Check experiment status
python scripts/experiments/swinv2_hybrid/analyze_360x360_experiment.py artifacts/experiments/YYYYMMDD-HHMMSS-swinv2_hybrid_360x360
```

### Analysis Tools

#### Memory Analysis

```python
# Analyze memory usage patterns
analyzer = ExperimentAnalyzer(experiment_dir)
memory_analysis = analyzer.analyze_memory_usage()
```

#### Performance Comparison

```python
# Compare with baseline models
comparison = analyzer.compare_with_baselines()
```

#### Training Progress

```python
# Analyze convergence and overfitting
progress = analyzer.analyze_training_progress()
```

## Expected Performance

### Target Metrics

- **IoU**: > 0.85
- **F1 Score**: > 0.90
- **Dice Coefficient**: > 0.88

### Baseline Comparison

| Model | IoU | F1 | Dice |
|-------|-----|----|----|
| U-Net | 0.681 | 0.811 | 0.792 |
| DeepLabV3+ | 0.688 | 0.816 | 0.799 |
| DeepCrack | 0.869 | 0.930 | 0.918 |
| **Our Target** | **>0.85** | **>0.90** | **>0.88** |

## Optimization Strategies

### Memory Optimization

1. **Gradient Accumulation**: Effective batch size 16 with batch size 2
2. **Mixed Precision**: AMP reduces memory usage by ~30%
3. **Gradient Checkpointing**: Trades compute for memory
4. **Memory Monitoring**: Continuous monitoring prevents OOM

### Training Optimization

1. **Early Stopping**: Patience 20 epochs, monitor val_iou
2. **Learning Rate**: 1e-4 with cosine annealing
3. **Data Augmentation**: Optimized for crack detection
4. **Loss Function**: FocalDiceLoss for class imbalance

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```bash
# Reduce batch size
batch_size: 1  # Instead of 2

# Increase gradient accumulation
gradient_accumulation_steps: 16  # Instead of 8

# Enable more aggressive memory optimization
memory:
  gradient_checkpointing: true
  empty_cache_freq: 50  # More frequent cleanup
```

#### Slow Training

```bash
# Increase batch size if memory allows
batch_size: 4  # If VRAM usage < 80%

# Reduce gradient accumulation
gradient_accumulation_steps: 4  # If batch_size increased

# Optimize data loading
dataloader:
  num_workers: 8  # Increase if CPU cores available
  prefetch_factor: 4
```

#### Poor Convergence

```bash
# Adjust learning rate
learning_rate: 0.00005  # Reduce if unstable

# Modify loss weights
loss:
  focal_weight: 0.7  # Increase focal loss
  dice_weight: 0.3   # Decrease dice loss

# Increase patience
early_stopping_patience: 30  # More patience
```

## Results Analysis

### Generated Files

After training, the experiment directory contains:

- `config.yaml`: Complete experiment configuration
- `final_results.yaml`: Final metrics
- `training.log`: Training progress log
- `analysis_report.txt`: Comprehensive analysis
- `checkpoints/`: Model checkpoints
- `predictions/`: Validation predictions
- `visualizations/`: Training visualizations

### Analysis Commands

```bash
# Generate analysis report
python scripts/experiments/swinv2_hybrid/analyze_360x360_experiment.py artifacts/experiments/YYYYMMDD-HHMMSS-swinv2_hybrid_360x360

# Compare with other experiments
python scripts/experiments/swinv2_hybrid/compare_experiments.py --experiments exp1 exp2 exp3

# Generate performance plots
python scripts/experiments/swinv2_hybrid/plot_results.py artifacts/experiments/YYYYMMDD-HHMMSS-swinv2_hybrid_360x360
```

## Best Practices

### Memory Management

1. Monitor GPU memory usage continuously
2. Use gradient accumulation for large images
3. Enable mixed precision training
4. Clear cache periodically

### Training Stability

1. Use reproducible seeds
2. Monitor loss curves for overfitting
3. Validate frequently (every epoch)
4. Save best model based on val_iou

### Performance Optimization

1. Use persistent workers in DataLoader
2. Enable pin_memory for GPU training
3. Optimize data augmentation pipeline
4. Monitor training/validation curves

## Expected Timeline

### Training Duration

- **Epochs**: 100 maximum
- **Early Stopping**: Usually 30-50 epochs
- **Time per Epoch**: ~15-20 minutes
- **Total Training Time**: 8-12 hours

### Checkpoints

- **Save Frequency**: Every 5 epochs
- **Best Model**: Based on val_iou
- **Checkpoint Size**: ~110MB (model + optimizer)

## Validation Strategy

### Metrics

- **IoU**: Intersection over Union
- **F1 Score**: Harmonic mean of precision/recall
- **Dice**: Similar to F1, good for class imbalance
- **Precision**: True positives / (true + false positives)
- **Recall**: True positives / (true + false negatives)

### Evaluation Frequency

- **Validation**: Every epoch
- **Metrics**: IoU, F1, Dice, Precision, Recall
- **Visualization**: Sample predictions saved

## Reproducibility

### Seed Configuration

```yaml
seed: 42
random_seed: 42
cudnn_deterministic: true
```

### Environment

- Python 3.12+
- PyTorch 2.0+
- CUDA 11.8+
- All dependencies pinned

### Checkpointing

- Complete model state
- Optimizer state
- Training metrics
- Configuration snapshot

## Future Improvements

### Potential Optimizations

1. **Larger Models**: SwinV2-base or SwinV2-large
2. **Advanced Loss**: Boundary-aware losses
3. **Ensemble Methods**: Multiple model predictions
4. **Post-processing**: CRF or morphological operations

### Research Directions

1. **Attention Visualization**: Analyze attention maps
2. **Feature Analysis**: Study encoder features
3. **Ablation Studies**: Component importance
4. **Transfer Learning**: Pre-trained on larger datasets

## References

- **SwinV2 Paper**: Liu et al. "Swin Transformer V2: Scaling Up Capacity and Resolution"
- **ASPP Module**: Chen et al. "DeepLab: Semantic Image Segmentation"
- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection"
- **Crack Segmentation**: Liu et al. "DeepCrack: Learning Hierarchical Convolutional Features"

## Contact

For questions or issues with this experiment:

- Check the training logs for error messages
- Review the analysis report for recommendations
- Consult the troubleshooting section above
- Open an issue in the project repository
