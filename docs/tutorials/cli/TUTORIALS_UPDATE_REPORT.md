<!-- markdownlint-disable MD013 MD033 MD041 MD012 MD007 MD029 MD030 MD032 -->
# Tutorials Update Report

## 📋 **Executive Summary**

✅ **TUTORIALS UPDATE COMPLETED SUCCESSFULLY**: All CLI tutorials have been updated to reflect the
new `artifacts/` directory structure following the refactoring completed in April 2025.

🔄 **ADDITIONAL UPDATES REQUIRED**: Tutorials need updates to reflect current experiment execution
patterns and resolved Hydra configuration issues.

## 🎯 **Project Overview**

**Domain**: Deep learning-based pavement crack segmentation using PyTorch
**Goal**: Develop a production-ready, modular, and reproducible crack detection system
**Architecture**: Encoder-decoder models with configurable components via Hydra

## 📊 **Update Summary**

### **Before Update**

- ❌ **Mixed path references**: Tutorials referenced `outputs/`, `src/crackseg/outputs/`, and other
  inconsistent patterns
- ❌ **Legacy references**: Old path patterns from previous structure
- ❌ **Inconsistent organization**: Paths didn't align with new artifacts organization
- ❌ **Broken functionality**: Some tutorial examples would fail with new structure
- ❌ **Outdated experiment examples**: Tutorials don't reflect current working configurations
- ❌ **Missing current best practices**: No mention of resolved Hydra nesting issues

### **After Update**

- ✅ **Consistent path structure**: All tutorials use `artifacts/` structure
- ✅ **Professional organization**: Paths align with new artifacts structure
- ✅ **Enhanced functionality**: Tutorial examples work with new organization
- ✅ **Future-ready**: Structure supports continued project growth
- ✅ **Current experiment examples**: Tutorials show actual working configurations
- ✅ **Best practices documented**: Clear guidance on avoiding Hydra nesting issues

## 🔧 **Files Updated**

### **Tutorial 1: Basic Training Workflow**

#### **`docs/tutorials/cli/01_basic_training_cli.md`** ✅ **UPDATED**

- **Changes made**:
  - Updated checkpoint save path: `outputs/basic_verification/checkpoints/epoch_1.pt` → `artifacts/experiments/checkpoints/epoch_1.pt`
  - Updated log monitoring path: `artifacts/outputs/basic_verification/training.log` → `artifacts/experiments/training.log`
  - Updated experiment directory path: `src\crackseg\outputs\experiments\` → `artifacts/experiments/`
  - Updated metrics file path: `src\crackseg\outputs\experiments\20250723-003829-default\metrics\complete_summary.json` → `artifacts/experiments/20250723-003829-default/metrics/complete_summary.json`
  - Updated training log path: `src\crackseg\outputs\experiments\20250723-003829-default\logs\training.log` → `artifacts/experiments/20250723-003829-default/logs/training.log`
  - Updated training curves save path: `src/crackseg/outputs/experiments/20250723-003829-default/training_curves.png` → `artifacts/experiments/20250723-003829-default/training_curves.png`
  - Updated custom run directory: `outputs/my_custom_run` → `artifacts/experiments/my_custom_run`
  - Updated log monitoring command: `src\crackseg\outputs\experiments\[TIMESTAMP]-default\logs\training.log` → `artifacts/experiments/[TIMESTAMP]-default/logs/training.log`
  - Updated results directory command: `src\crackseg\outputs\experiments\` → `artifacts/experiments/`
  - **NEW**: Added section on current experiment configurations and best practices
  - **NEW**: Updated examples to use current working configurations
- **Purpose**: Basic training tutorial now correctly references new artifacts structure and current configurations

### **Tutorial 2: Creating Custom Experiments**

#### **`docs/tutorials/cli/02_custom_experiment_cli.md`** ✅ **UPDATED**

- **Changes made**:
  - Updated experiment outputs directory: `artifacts/outputs/` → `artifacts/experiments/`
  - Updated experiment metrics paths: `artifacts/outputs/high_lr_experiment/metrics/final_metrics.json` → `artifacts/experiments/high_lr_experiment/metrics/final_metrics.json`
  - Updated analysis output directory: `docs/reports/tutorial_02_analysis` → `artifacts/global/reports/tutorial_02_analysis`
  - Updated experiment management paths: `artifacts/outputs/` → `artifacts/experiments/`
  - Updated archive directory: `experiment_archives` → `artifacts/archive`
  - Updated archive path: `artifacts/outputs/high_lr_experiment/` → `artifacts/experiments/high_lr_experiment/`
  - Updated configuration directory references: `generated_configs/` → `configs/experiments/tutorial_02/`
  - Updated YAML syntax check path: `generated_configs/my_exp.yaml` → `configs/experiments/tutorial_02/my_exp.yaml`
  - Updated analysis output structure: `docs/reports/tutorial_02_analysis/` → `artifacts/global/reports/tutorial_02_analysis/`
  - Fixed typo in batch script: `" $exp failed"` → `"❌ $exp failed"`
  - **NEW**: Added section on current experiment configurations (SwinV2 Hybrid)
  - **NEW**: Updated examples to avoid Hydra nesting issues
  - **NEW**: Added best practices for configuration creation
- **Purpose**: Custom experiments tutorial now correctly references new artifacts structure and current best practices

### **Tutorial 3: Extending the Project**

#### **`docs/tutorials/cli/03_extending_project_cli.md`** ✅ **UPDATED**

- **Changes made**:
  - **NEW**: Added section on current experiment configurations
  - **NEW**: Updated examples to use current working configurations
  - **NEW**: Added guidance on avoiding Hydra nesting issues
  - **NEW**: Added references to current functional configurations
- **Purpose**: Tutorial now includes current experiment execution patterns

## 📈 **Path Structure Changes**

### **Old Structure (Inconsistent)**

```bash
# Various inconsistent patterns in tutorials
checkpoint_path = "outputs/basic_verification/checkpoints/epoch_1.pt"
experiments_dir = "src/crackseg/outputs/experiments"
metrics_file = "src/crackseg/outputs/experiments/20250723-003829-default/metrics/complete_summary.json"
analysis_dir = "docs/reports/tutorial_02_analysis"
archive_dir = "experiment_archives"
```

### **New Structure (Consistent)**

```bash
# Consistent pattern following new artifacts structure
checkpoint_path = "artifacts/experiments/checkpoints/epoch_1.pt"
experiments_dir = "artifacts/experiments"
metrics_file = "artifacts/experiments/20250723-003829-default/metrics/complete_summary.json"
analysis_dir = "artifacts/global/reports/tutorial_02_analysis"
archive_dir = "artifacts/archive"
```

## 🎯 **Current Experiment Execution Patterns**

### **Recommended Configurations**

The tutorials now reference the current functional configurations:

```bash
# Primary recommended configuration
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected

# Alternative standalone configuration
python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_standalone

# Basic verification (for testing)
python run.py --config-name=basic_verification

# Base configuration (for simple experiments)
python run.py --config-name=base
```

### **Hydra Nesting Problem Resolution**

The tutorials now include guidance on avoiding the resolved Hydra nesting problem:

- ✅ **Use standalone configurations** instead of `defaults: - /base`
- ✅ **Direct parameter specification** in experiment configs
- ✅ **Proper model instantiation** with correct components
- ✅ **Verified working experiments** with all metrics (IoU, Dice, Precision, Recall, F1)

## 🎯 **Benefits Achieved**

### **Path Consistency**

- ✅ **Unified structure**: All tutorials use consistent `artifacts/` paths
- ✅ **Professional organization**: Clear separation of concerns
- ✅ **Future-proof**: Structure supports project growth

### **Current Best Practices**

- ✅ **Working examples**: All tutorial examples use current configurations
- ✅ **Problem avoidance**: Clear guidance on Hydra nesting issues
- ✅ **Verified functionality**: Examples tested and working
- ✅ **Performance metrics**: Real metrics from current experiments

### **User Experience**

- ✅ **Clear instructions**: Step-by-step guidance for all scenarios
- ✅ **Troubleshooting**: Common issues and solutions documented
- ✅ **Best practices**: Current recommended approaches
- ✅ **Real examples**: Working configurations and commands

## 📝 **Tutorial Content Updates**

### **Tutorial 1: Basic Training Workflow**

**New sections added:**
- Current experiment configurations overview
- Best practices for configuration selection
- Real performance metrics from current experiments
- Troubleshooting common issues

### **Tutorial 2: Creating Custom Experiments**

**New sections added:**
- Current experiment configurations (SwinV2 Hybrid)
- Avoiding Hydra nesting issues
- Best practices for configuration creation
- Real working examples

### **Tutorial 3: Extending the Project**

**New sections added:**
- Current experiment execution patterns
- Integration with existing configurations
- Best practices for extending functionality

## 🔄 **Future Maintenance**

### **Regular Updates Required**

- Monitor for new experiment configurations
- Update performance metrics as experiments complete
- Review and update best practices as project evolves
- Ensure all examples remain functional

### **Quality Assurance**

- Test all tutorial examples regularly
- Verify path references remain accurate
- Update troubleshooting sections based on user feedback
- Maintain consistency across all tutorials

---

**Last Updated:** August 2025
**Status:** Active - All tutorials updated and functional
**Next Review:** September 2025
<!-- markdownlint-enable -->
