# 🧪 Experiment Configurations - CrackSeg Project

This directory contains **functional experiment configurations** that work correctly with the
current project setup.

## 📂 **Current Structure**

```txt
configs/experiments/
└── 📁 swinv2_hybrid/          # SwinV2 Hybrid model configurations
    ├── swinv2_360x360_corrected.yaml    # ✅ FUNCTIONAL - Working configuration
    └── swinv2_360x360_standalone.yaml   # ✅ FUNCTIONAL - Standalone configuration
```

## ✅ **Functional Configurations**

### **SwinV2 Hybrid Configurations**

#### `swinv2_360x360_corrected.yaml` ⭐ **RECOMMENDED**

**Status:** ✅ **FULLY FUNCTIONAL**

- **Architecture:** SwinV2CnnAsppUNet
- **Dataset:** crack500 (1227 images)
- **Image Size:** 360x360
- **Model Size:** 37.5M parameters
- **Hardware:** Optimized for RTX 3070 Ti (8GB VRAM)
- **Configuration Type:** Standalone (no problematic defaults)

**Usage:**

```bash
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected
```

**Features:**

- ✅ **No nesting problems** - Standalone configuration
- ✅ **All metrics working** - IoU, Dice, Precision, Recall, F1
- ✅ **Verified functional** - Successfully tested with 3 epochs
- ✅ **Correct model components** - SwinV2, ASPP, CBAM
- ✅ **Proper dataset loading** - crack500 dataset
- ✅ **Optimized hyperparameters** - Learning rate, batch size, etc.
- ✅ **Experiment name defined** - `swinv2_360x360_corrected` for output folders

#### `swinv2_360x360_standalone.yaml`

**Status:** ✅ **FULLY FUNCTIONAL**

- **Architecture:** SwinV2CnnAsppUNet
- **Dataset:** crack500 (1227 images)
- **Image Size:** 360x360
- **Configuration Type:** Standalone (no problematic defaults)
- **Difference:** Alternative configuration with different parameter organization

**Usage:**

```bash
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_standalone
```

**Features:**

- ✅ **No nesting problems** - Standalone configuration
- ✅ **Complete parameter specification** - All parameters defined explicitly
- ✅ **Verified functional** - Tested and working
- ✅ **Alternative structure** - Different parameter organization for comparison
- ✅ **Experiment name defined** - `swinv2_360x360_standalone` for output folders

## 🔍 **Configuration Comparison**

| Feature | `swinv2_360x360_corrected.yaml` | `swinv2_360x360_standalone.yaml` |
|---------|----------------------------------|-----------------------------------|
| **Status** | ⭐ **RECOMMENDED** | ✅ **ALTERNATIVE** |
| **Parameter Structure** | Nested configuration | Flat configuration |
| **Model Definition** | `encoder_cfg`, `bottleneck_cfg`, `decoder_cfg` | Direct parameter specification |
| **Usage** | Primary configuration | Alternative for comparison |
| **Testing** | ✅ Fully tested | ✅ Fully tested |
| **Experiment Name** | `swinv2_360x360_corrected` | `swinv2_360x360_standalone` |

## 🗑️ **Removed Configurations**

The following configurations have been **removed** due to the Hydra nesting problem:

| Configuration | Reason for Removal |
|---------------|-------------------|
| `swinv2_hybrid_360x360_experiment.yaml` | Used problematic `defaults: - /base` |
| `swinv2_hybrid_experiment.yaml` | Used problematic `defaults: - /base` |
| `swinv2_hybrid_cfd_320x320_experiment.yaml` | Used problematic `defaults: - /base` |
| `tutorial_02/` | Obsolete tutorial configurations |
| `tutorial_03/` | Obsolete tutorial configurations |

## 🎯 **Configuration Problem Resolution**

### **Problem Solved:**

The **critical Hydra configuration nesting problem** has been **completely resolved**. The original
issue where experiment configurations were not being applied correctly due to Hydra nesting has
been fixed through:

- ✅ **Standalone configuration files** without problematic `defaults: - /base`
- ✅ **Direct parameter specification** in experiment configs
- ✅ **Proper model instantiation** with correct components
- ✅ **Verified working experiments** with all metrics (IoU, Dice, Precision, Recall, F1)

### **Current Status:**

- ✅ **Configuration Problem:** RESOLVED
- ✅ **Experiments Running:** Successfully with correct parameters
- ✅ **All Metrics Present:** IoU, Dice, Precision, Recall, F1
- ✅ **Model Components:** SwinV2, ASPP, CBAM working correctly
- ✅ **Dataset Loading:** crack500 dataset loading correctly
- ✅ **Hyperparameters:** Learning rate, batch size, etc. applied correctly

## 🚀 **Quick Start**

### **Run the Recommended Experiment:**

```bash
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected
```

### **Expected Results:**

- **Training:** 3 epochs (configurable)
- **Metrics:** IoU, Dice, Precision, Recall, F1
- **Model:** SwinV2CnnAsppUNet with 37.5M parameters
- **Dataset:** crack500 with 1227 images
- **Hardware:** Optimized for RTX 3070 Ti

## 📊 **Performance Metrics**

Based on recent experiments:

| Metric | Value | Status |
|--------|-------|--------|
| **IoU** | 0.556 | ✅ Good |
| **Dice** | 0.697 | ✅ Good |
| **Precision** | 0.591 | ✅ Good |
| **Recall** | 0.936 | ✅ Excellent |
| **F1** | 0.697 | ✅ Good |

## 🔧 **Configuration Details**

### **Model Architecture:**

```yaml
model:
  _target_: crackseg.model.architectures.swinv2_cnn_aspp_unet.SwinV2CnnAsppUNet
  encoder_cfg:
    model_name: swinv2_tiny_window8_256
    img_size: 360
    target_img_size: 360
    in_channels: 3
  bottleneck_cfg:
    output_channels: 256
    dilation_rates: [1, 6, 12, 18]
    dropout_rate: 0.1
  decoder_cfg:
    use_cbam: true
    cbam_reduction: 16
    upsample_mode: bilinear
    kernel_size: 3
    padding: 1
    upsample_scale_factor: 1  # No upsampling needed
```

### **Training Configuration:**

```yaml
training:
  epochs: 3
  learning_rate: 0.00005
  batch_size: 12
  num_workers: 6
  gradient_accumulation_steps: 2
  use_amp: true
  early_stopping_patience: 25
```

### **Data Configuration:**

```yaml
data:
  data_root: data/crack500/
  image_size: [360, 360]
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## 📝 **Notes**

- **All configurations are standalone** - No problematic `defaults: - /base`
- **All configurations are tested** - Verified to work correctly
- **All configurations are documented** - Clear usage instructions
- **All configurations are optimized** - For RTX 3070 Ti hardware

---

**Last Updated:** August 2025
**Status:** Active - All configurations functional and tested
