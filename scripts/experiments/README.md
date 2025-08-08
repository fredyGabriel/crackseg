# 🧪 Experiment Scripts - CrackSeg Project

This directory contains organized scripts to run, verify, analyze and debug crack segmentation experiments.

## 📂 Organized Structure

```bash
scripts/experiments/
├── 📁 analysis/               # Experiment analysis
├── 📁 benchmarking/           # Comparisons and benchmarks
├── 📁 demos/                  # Examples and demonstrations
├── 📁 debugging/              # Debugging tools
├── 📁 e2e/                    # End-to-end tests
├── 📁 tutorials/              # Educational tutorials
└── 📄 README.md               # This documentation
```

---

## ✅ **CONFIGURATION PROBLEM RESOLVED**

### 🎉 **Problem Status: RESOLVED**

The **critical configuration nesting problem** has been **completely resolved**. The original issue
where experiment configurations were not being applied correctly due to Hydra nesting has been
fixed through:

- ✅ **Standalone configuration files** without problematic `defaults: - /base`
- ✅ **Direct parameter specification** in experiment configs
- ✅ **Proper model instantiation** with correct components
- ✅ **Verified working experiments** with all metrics (IoU, Dice, Precision, Recall, F1)

### 🗑️ **Obsolete Scripts Removed**

The following scripts have been **removed** as they are no longer needed:

- ❌ `runners/` - All fixed runner scripts (problem resolved)
- ❌ `config-verification/` - All verification scripts (problem resolved)

### 🚀 **Current Usage**

Experiments now run directly with the standard command:

```bash
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected
```

---

## 🔧 **Tools by Category**

### 📊 **Analysis** - Experiment Analysis

#### `analysis/swinv2_hybrid/analysis/analyze_experiment.py`

Comprehensive experiment analysis tool with memory usage, training progress, and performance benchmarking.

```bash
# Analyze a specific experiment
python scripts/experiments/analysis/swinv2_hybrid/analysis/analyze_experiment.py \
    --experiment-dir artifacts/experiments/20250806-213037-default_experiment
```

**Features:**

- ✅ Memory usage analysis and optimization recommendations
- ✅ Training progress monitoring
- ✅ Performance benchmarking
- ✅ Hardware utilization analysis
- ✅ Experiment comparison tools
- ✅ Dataset-specific analysis

---

### ⚖️ **Benchmarking** - Comparisons and Performance

#### `benchmarking/automated_comparison.py`

Automated comparison of multiple experiments with comprehensive analysis reports.

```bash
# Compare multiple experiments
python scripts/experiments/benchmarking/automated_comparison.py \
    --experiments exp1,exp2,exp3
```

#### `benchmarking/benchmark_aspp.py`

ASPP module performance benchmarking and analysis.

---

### 🎯 **Demos** - Examples and Demonstrations

#### `demos/registry_demo.py`

Basic registry demonstration showing component registration.

#### `demos/hybrid_registry_demo.py`

Advanced hybrid registry demonstration with complex configurations.

#### `demos/example_generalized_experiment.py`

Generalized experiment example for different datasets and configurations.

---

### 🐛 **Debugging** - Debugging Tools

#### `debugging/debug_swin_params.py`

SwinV2 parameter debugging and analysis tool.

---

### 🧪 **E2E** - End-to-End Tests

#### `e2e/test_pipeline_e2e.py`

Complete pipeline testing with synthetic datasets and reduced models.

```bash
# Run end-to-end test
python scripts/experiments/e2e/test_pipeline_e2e.py
```

**Features:**

- ✅ Complete pipeline verification
- ✅ Checkpoint saving/loading
- ✅ Model evaluation
- ✅ Results reporting

---

### 📚 **Tutorials** - Educational Tutorials

#### `tutorials/tutorial_02/`

Educational tutorial scripts for learning experiment workflows.

- `tutorial_02_compare.py` - Experiment comparison tutorial
- `tutorial_02_visualize.py` - Visualization tutorial
- `tutorial_02_batch.ps1` - Batch processing tutorial

---

## 🎯 **Quick Start**

### 1. **Run an Experiment**

```bash
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected
```

### 2. **Analyze Results**

```bash
python scripts/experiments/analysis/swinv2_hybrid/analysis/analyze_experiment.py \
    --experiment-dir artifacts/experiments/[EXPERIMENT_NAME]
```

### 3. **Compare Experiments**

```bash
python scripts/experiments/benchmarking/automated_comparison.py \
    --auto-find --max-experiments 5
```

### 4. **Run E2E Test**

```bash
python scripts/experiments/e2e/test_pipeline_e2e.py
```

---

## 📋 **File Organization**

### **Maintained Scripts:**

| Category | Script | Purpose |
|----------|--------|---------|
| **Analysis** | `analyze_experiment.py` | Comprehensive experiment analysis |
| **Benchmarking** | `automated_comparison.py` | Multi-experiment comparison |
| **Benchmarking** | `benchmark_aspp.py` | ASPP module benchmarking |
| **Demos** | `registry_demo.py` | Basic registry demonstration |
| **Demos** | `hybrid_registry_demo.py` | Advanced registry demo |
| **Demos** | `example_generalized_experiment.py` | Generalized experiment example |
| **Debugging** | `debug_swin_params.py` | SwinV2 parameter debugging |
| **E2E** | `test_pipeline_e2e.py` | Complete pipeline testing |
| **Tutorials** | `tutorial_02_*.py` | Educational tutorials |

### **Removed Scripts (Obsolete):**

| Script | Reason |
|--------|--------|
| `run_swinv2_experiment_fixed.py` | Configuration problem resolved |
| `verify_config_simple.py` | Configuration problem resolved |
| `analyze_config_problem.py` | Configuration problem resolved |
| All other fixed runners | Configuration problem resolved |

---

## 🎉 **Success Metrics**

- ✅ **Configuration Problem:** RESOLVED
- ✅ **Experiments Running:** Successfully with correct parameters
- ✅ **All Metrics Present:** IoU, Dice, Precision, Recall, F1
- ✅ **Model Components:** SwinV2, ASPP, CBAM working correctly
- ✅ **Dataset Loading:** crack500 dataset loading correctly
- ✅ **Hyperparameters:** Learning rate, batch size, etc. applied correctly

---

**Last Updated:** August 2025
**Status:** Active - Configuration problem resolved, scripts cleaned up
