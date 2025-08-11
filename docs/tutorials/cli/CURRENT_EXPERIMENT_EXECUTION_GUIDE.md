# 🚀 Current Experiment Execution Guide - CrackSeg Project

## 📋 **Executive Summary**

This guide provides **current, verified instructions** for executing experiments in the CrackSeg
project, reflecting the latest project state after resolving the Hydra configuration nesting problem.

## 🎯 **Current Working Configurations**

### **Primary Recommended Configurations**

```bash
# Primary recommended configuration (SwinV2 Hybrid)
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected

# Alternative standalone configuration
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_standalone
```

### **Testing Configurations**

```bash
# Basic verification (for testing)
python run.py --config-name=basic_verification

# Base configuration (for simple experiments)
python run.py --config-name=base
```

## 📊 **Performance Metrics**

Based on recent experiments with the current configurations:

| Metric | Value | Status |
|--------|-------|--------|
| **IoU** | 0.556 | ✅ Good |
| **Dice** | 0.697 | ✅ Good |
| **Precision** | 0.591 | ✅ Good |
| **Recall** | 0.936 | ✅ Excellent |
| **F1** | 0.697 | ✅ Good |

## 🔧 **Configuration Details**

### **SwinV2 Hybrid Configurations**

#### `swinv2_360x360_corrected.yaml` ⭐ **RECOMMENDED**

- **Architecture:** SwinV2CnnAsppUNet
- **Dataset:** crack500 (1227 images)
- **Image Size:** 360x360
- **Model Size:** 37.5M parameters
- **Hardware:** Optimized for RTX 3070 Ti (8GB VRAM)
- **Configuration Type:** Standalone (no problematic defaults)

#### `swinv2_360x360_standalone.yaml`

- **Architecture:** SwinV2CnnAsppUNet
- **Dataset:** crack500 (1227 images)
- **Image Size:** 360x360
- **Configuration Type:** Standalone (no problematic defaults)
- **Difference:** Alternative configuration with different parameter organization

## ✅ **Best Practices**

### **Configuration Creation**

- ✅ **Use standalone configurations** instead of `defaults: - /base`
- ✅ **Specify all parameters explicitly** to avoid Hydra nesting issues
- ✅ **Test configurations** before running full experiments
- ✅ **Use current working configurations** as templates

### **Execution Commands**

- ✅ **Use `run.py`** instead of `src/main.py` for proper execution
- ✅ **Activate conda environment** before running: `conda activate crackseg`
- ✅ **Use correct config paths** for experiment configurations
- ✅ **Monitor training progress** with real-time logs

## 🚫 **Avoid These Issues**

### **Hydra Nesting Problem (RESOLVED)**

The following configurations have been **removed** due to the Hydra nesting problem:

| Configuration | Reason for Removal |
|---------------|-------------------|
| `swinv2_hybrid_360x360_experiment.yaml` | Used problematic `defaults: - /base` |
| `swinv2_hybrid_experiment.yaml` | Used problematic `defaults: - /base` |
| `swinv2_hybrid_cfd_320x320_experiment.yaml` | Used problematic `defaults: - /base` |

### **Common Mistakes**

- ❌ **Don't use `defaults: - /base`** in new configurations
- ❌ **Don't use `src/main.py`** - use `run.py` instead
- ❌ **Don't forget to activate conda environment** before running
- ❌ **Don't use obsolete configurations** that were removed

## 📁 **Project Structure**

### **Current Configuration Structure**

```txt
configs/
├── base.yaml                    # ✅ Working base configuration
├── basic_verification.yaml      # ✅ Working verification configuration
└── experiments/
    └── swinv2_hybrid/
        ├── swinv2_360x360_corrected.yaml    # ✅ RECOMMENDED
        └── swinv2_360x360_standalone.yaml   # ✅ ALTERNATIVE
```

### **Output Structure**

```txt
artifacts/
└── experiments/
    └── [TIMESTAMP]-[CONFIG_NAME]/
        ├── checkpoints/         # Model checkpoints
        ├── logs/               # Training logs
        ├── metrics/            # Evaluation metrics
        └── configurations/     # Configuration snapshots
```

## 🔍 **Monitoring and Debugging**

### **Real-Time Monitoring**

```bash
# Monitor training progress
tail -f artifacts/experiments/[TIMESTAMP]-[CONFIG_NAME]/logs/training.log

# Check experiment status
dir artifacts/experiments/
```

### **Troubleshooting**

#### **Common Issues and Solutions**

1. **Configuration Not Found**

   ```bash
   # Solution: Use correct config path
   python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected
   ```

2. **Import Errors**

   ```bash
   # Solution: Install package in development mode
   pip install -e . --no-deps
   ```

3. **CUDA Out of Memory**

   ```bash
   # Solution: Reduce batch size
   python run.py --config-name=basic_verification dataloader.batch_size=4
   ```

4. **Hydra Nesting Issues**

   ```bash
   # Solution: Use standalone configurations
   # Avoid: defaults: - /base
   # Use: Complete parameter specification
   ```

## 📝 **Tutorial Integration**

### **Updated Tutorials**

The following tutorials have been updated to reflect current best practices:

- ✅ **Tutorial 1**: Basic Training Workflow - Updated with current configurations
- ✅ **Tutorial 2**: Creating Custom Experiments - Updated with best practices
- ✅ **Tutorial 3**: Extending the Project - Updated with current patterns

### **Key Updates**

- ✅ **Current configuration examples** instead of obsolete ones
- ✅ **Best practices for avoiding Hydra nesting issues**
- ✅ **Real performance metrics** from current experiments
- ✅ **Working command examples** that have been tested

## 🎯 **Quick Reference**

### **Most Common Commands**

```bash
# Run recommended experiment
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected

# Run basic verification
python run.py --config-name=basic_verification

# Monitor training
tail -f artifacts/experiments/[TIMESTAMP]-[CONFIG_NAME]/logs/training.log

# Check results
dir artifacts/experiments/
```

### **Configuration Files**

| File | Purpose | Status |
|------|---------|--------|
| `swinv2_360x360_corrected.yaml` | Primary production | ✅ **RECOMMENDED** |
| `swinv2_360x360_standalone.yaml` | Alternative production | ✅ **FUNCTIONAL** |
| `basic_verification.yaml` | Testing | ✅ **WORKING** |
| `base.yaml` | Simple experiments | ✅ **WORKING** |

---

**Last Updated:** August 2025
**Status:** Active - All configurations tested and functional
**Next Review:** September 2025
