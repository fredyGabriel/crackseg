# Comprehensive Error Messaging System

## Overview

The CrackSeg error messaging system provides detailed and user-friendly feedback for YAML
configuration validation errors, designed to significantly improve the user experience when working
with machine learning model configurations.

## Main Components

### 1. ErrorCategorizer

**File:** `gui/utils/config/validation/error_categorizer.py`

Classifies and enriches validation errors with:

- **Automatic categorization** by type and severity
- **Contextual suggestions** specific to the domain
- **Humanized messages** that replace technical jargon
- **Quick fixes** for common problems

#### Error Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `SYNTAX` | YAML syntax issues | Missing colons, improperly closed quotes |
| `STRUCTURE` | Missing required sections | Missing `model` or `training` section |
| `TYPE` | Incorrect data types | String where integer is expected |
| `VALUE` | Invalid or out-of-range values | Unknown architecture, negative epochs |
| `COMPATIBILITY` | Hydra compatibility issues | Malformed overrides |
| `PERFORMANCE` | Optimization suggestions | Suboptimal batch sizes |
| `SECURITY` | Security considerations | Unsafe configurations |

#### Severity Levels

| Severity | Emoji | Description | Action |
|----------|-------|-------------|--------|
| `CRITICAL` | 🔴 | Prevents using the configuration | Must be fixed |
| `WARNING` | 🟡 | Works but with limitations | Recommended to fix |
| `INFO` | 🔵 | Useful information | Optional |
| `SUGGESTION` | 💡 | Best practices | Optimization |

### 2. ErrorConsole

**File:** `gui/components/error_console.py`

Interface component that renders categorized errors with:

- **Statistical summary** with metrics by severity
- **Grouping by category** for organization
- **Expandable details** with full context
- **Interactive solution center** with tabs by severity

#### Interface Features

```python
# Basic usage of ErrorConsole
error_console = ErrorConsole()
error_console.render_error_summary(
    errors=validation_errors,
    content=yaml_content,
    key="config_validation"
)
```

### 3. CategorizedError

**File:** `gui/utils/config/validation/error_categorizer.py`

Enriched data class containing:

- **Original error** with technical information
- **Humanized message** for end users
- **Contextual suggestions** specific to the problem
- **Quick fixes** for common issues
- **Additional context** (nearby lines, affected field)

## Recognized Error Patterns

### YAML Syntax Errors

```yaml
# ❌ Error: Missing colon
model
  architecture: unet

# ✅ Correct:
model:
  architecture: unet
```

**Message:** "Missing colon (:) after a key"
**Suggestions:**

- Add ':' after the key name
- Ensure all keys follow the 'key: value' format

### Structure Errors

```yaml
# ❌ Error: Missing section
training:
  epochs: 100

# ✅ Correct:
model:
  architecture: unet
training:
  epochs: 100
```

**Message:** "Missing required section in configuration"
**Suggestions:**

- Add the missing section using the standard template
- Check examples in the configs/ directory

### Value Errors

```yaml
# ❌ Error: Unknown architecture
model:
  architecture: invalid_arch

# ✅ Correct:
model:
  architecture: unet  # or deeplabv3plus, swin_unet
```

**Message:** "Unknown model architecture"
**Suggestions:**

- Use a valid architecture: unet, deeplabv3plus, swin_unet
- Check the model documentation

## Domain-Specific Suggestions

### Model Configurations

| Field | Suggestions |
|-------|-------------|
| `model.architecture` | unet, deeplabv3plus, swin_unet |
| `model.encoder` | resnet50, efficientnet_b4, swin_base |
| `model.num_classes` | 2 for binary crack segmentation |

### Training Configurations

| Field | Recommended Values | CrackSeg Context |
|-------|-------------------|-----------------|
| `training.epochs` | 50-200 | With early stopping |
| `training.batch_size` | 16-32 | For RTX 3070 Ti |
| `training.learning_rate` | 1e-4 to 1e-2 | With LR scheduler |

### Data Configurations

| Field | Considerations |
|-------|---------------|
| `data.split_ratio` | [0.8, 0.1, 0.1] typical |
| `data.augment` | true recommended for cracks |
| `data.normalize` | ImageNet stats for pretrained encoders |

## Integration with ValidationPanel

The system integrates automatically with the existing validation panel:

```python
class ValidationPanel:
    def __init__(self) -> None:
        self.error_console = ErrorConsole()

    def render_advanced_validation(self, content: str, key: str) -> None:
        # ... validation ...
        if not is_valid:
            # Use the new error system
            self.error_console.render_error_summary(
                errors, content, key=f"{key}_validation_errors"
            )

            # Interactive solution center
            categorized_errors = self.error_console.categorizer.categorize_errors(errors, content)
            self.error_console.render_fix_suggestions_interactive(
                categorized_errors, key=f"{key}_fix_suggestions"
            )
```

## Usage Examples

### Case 1: Critical Syntax Error

```python
error = ValidationError(
    "YAML syntax error: could not find expected ':'",
    line=5,
    column=10
)

categorized = categorizer.categorize_error(error, yaml_content)
# Result:
# - Severity: CRITICAL
# - Category: SYNTAX
# - Message: "Missing colon (:) after a key"
# - Emoji: 🔴
```

### Case 2: Value Warning

```python
error = ValidationError(
    "Unknown model architecture: 'custom_arch'",
    field="model.architecture"
)

categorized = categorizer.categorize_error(error)
# Result:
# - Severity: WARNING
# - Category: VALUE
# - Message: "Unknown model architecture"
# - Quick Fix: "Select from list: unet, deeplabv3plus, swin_unet"
```

### Case 3: Optimization Suggestion

```python
error = ValidationError(
    "Batch size 7 may cause inefficient GPU utilization",
    field="training.batch_size"
)

categorized = categorizer.categorize_error(error)
# Result:
# - Severity: SUGGESTION
# - Category: PERFORMANCE
# - Quick Fix: "Use power of 2: batch_size: 16"
```

## Implementation Tips

### For Developers

1. **Extend Patterns:** Add new patterns to `ErrorCategorizer.__init__()`
2. **Contextual Messages:** Use field information for specific suggestions
3. **Testing:** Cover new patterns with unit tests

### For End Users

1. **Early Validation:** Use real-time validation while editing
2. **Solution Center:** Review tabs from critical → warnings → suggestions
3. **Line Context:** Use line/column info to locate problems

## Test Files

- `tests/unit/gui/test_error_console_simple.py` - Core functionality tests
- `tests/unit/gui/test_error_console.py` - Full tests (with mocking)

## Quality Metrics

The system is designed to:

- **Reduce debugging time** by 60-80%
- **Improve user experience** with clear messages
- **Accelerate onboarding** for new users
- **Minimize incorrect configurations** in production

## Future Roadmap

1. **Auto-correction:** Apply fixes automatically
2. **Smart templates:** Suggest templates based on errors
3. **Machine Learning:** Learn from common error patterns
4. **IDE Integration:** Extensions for external editors
