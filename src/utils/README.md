# Utils Module

This directory contains utility functions and helper modules that support the core functionality of the pavement crack segmentation project. The utilities are organized by purpose and provide reusable components across the entire codebase.

## Directory Structure

```txt
src/utils/
├── checkpointing/          # Model checkpoint management
├── config/                 # Configuration utilities and validation
├── core/                   # Core utilities (device management, seeds, etc.)
├── experiment/             # Experiment tracking and management
├── factory/                # Factory patterns and component creation
├── logging/                # Logging setup and configuration
├── training/               # Training-specific utilities
├── visualization/          # Visualization and plotting utilities
├── component_cache.py      # Component caching system
├── exceptions.py           # Custom exception classes
└── __init__.py            # Module initialization and exports
```

## Module Overview

### Checkpointing (`checkpointing/`)

- **Purpose**: Model checkpoint saving, loading, and management
- **Key Features**:
  - Automatic checkpoint saving during training
  - Best model tracking based on validation metrics
  - Resume training from checkpoints
  - Checkpoint format validation and compatibility

### Configuration (`config/`)

- **Purpose**: Configuration management and validation utilities
- **Key Features**:
  - Hydra configuration validation
  - Configuration schema enforcement
  - Parameter interpolation and resolution
  - Configuration merging and overrides

### Core Utilities (`core/`)

- **Purpose**: Fundamental utilities used across the project
- **Key Features**:
  - Device management (CPU/GPU detection and setup)
  - Random seed control for reproducibility
  - Path management and validation
  - System resource monitoring

### Experiment Management (`experiment/`)

- **Purpose**: Experiment tracking and organization
- **Key Features**:
  - Experiment directory creation and management
  - Metadata tracking and storage
  - Result aggregation and comparison
  - Experiment configuration archival

### Factory Patterns (`factory/`)

- **Purpose**: Dynamic component creation and registration
- **Key Features**:
  - Generic factory pattern implementation
  - Component registry management
  - Dynamic instantiation from configuration
  - Type-safe component creation

### Logging (`logging/`)

- **Purpose**: Logging setup and configuration
- **Key Features**:
  - Structured logging configuration
  - Multi-level logging (console, file, remote)
  - Training progress logging
  - Performance metrics logging

### Training Utilities (`training/`)

- **Purpose**: Training-specific helper functions
- **Key Features**:
  - Early stopping implementation
  - Learning rate scheduling utilities
  - Training loop helpers
  - Validation utilities

### Visualization (`visualization/`)

- **Purpose**: Data and result visualization
- **Key Features**:
  - Training curve plotting
  - Segmentation mask visualization
  - Model architecture diagrams
  - Performance metric charts

## Key Components

### Component Cache (`component_cache.py`)

Provides caching mechanisms for expensive component operations:

```python
from src.utils.component_cache import ComponentCache

cache = ComponentCache()
cached_result = cache.get_or_compute(key, expensive_function, *args)
```

### Custom Exceptions (`exceptions.py`)

Defines project-specific exception classes:

```python
from src.utils.exceptions import ConfigurationError, ModelError

raise ConfigurationError("Invalid model configuration")
```

## Usage Examples

### Device Management

```python
from src.utils.core.device import get_device, set_device

device = get_device()  # Auto-detect best available device
set_device('cuda:0')   # Force specific device
```

### Seed Control

```python
from src.utils.core.seeds import set_seed

set_seed(42)  # Set deterministic seed for reproducibility
```

### Checkpoint Management

```python
from src.utils.checkpointing import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(model, optimizer, epoch, metrics, 'checkpoint.pth')

# Load checkpoint
checkpoint = load_checkpoint('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Logging Setup

```python
from src.utils.logging import setup_logger

logger = setup_logger('training', level='INFO', file_output=True)
logger.info("Training started")
```

### Early Stopping

```python
from src.utils.training.early_stopping import EarlyStopping

early_stopping = EarlyStopping(patience=10, min_delta=0.001)
if early_stopping(val_loss):
    print("Early stopping triggered")
    break
```

## Integration Points

The utils module integrates with all major components:

- **Training Pipeline**: Checkpointing, logging, early stopping
- **Data Pipeline**: Path management, configuration validation
- **Model Pipeline**: Device management, factory patterns
- **Evaluation Pipeline**: Visualization, result aggregation
- **Configuration System**: Hydra integration, validation

## Best Practices

1. **Modularity**: Each utility module has a single, well-defined purpose
2. **Type Safety**: All utilities include comprehensive type annotations
3. **Error Handling**: Robust error handling with custom exceptions
4. **Documentation**: All functions include detailed docstrings
5. **Testing**: Comprehensive unit tests for all utility functions
6. **Configurability**: Utilities accept configuration parameters when appropriate

## Extension Guidelines

When adding new utilities:

1. **Choose the Right Module**: Place utilities in the appropriate subdirectory
2. **Follow Patterns**: Use existing patterns for consistency
3. **Add Tests**: Include comprehensive unit tests
4. **Document**: Provide clear docstrings and usage examples
5. **Type Annotations**: Include complete type hints
6. **Error Handling**: Handle edge cases gracefully

## Dependencies

The utils module has minimal external dependencies:

- **PyTorch**: For device management and tensor operations
- **Hydra/OmegaConf**: For configuration management
- **Logging**: Standard library logging
- **Pathlib**: For path operations
- **Typing**: For type annotations

## Performance Considerations

- **Caching**: Use `component_cache.py` for expensive operations
- **Lazy Loading**: Import utilities only when needed
- **Memory Management**: Clean up resources in long-running utilities
- **Profiling**: Monitor performance of frequently used utilities

## Related Documentation

- **Configuration Guide**: `docs/guides/configuration_storage_specification.md`
- **Training Workflow**: `docs/guides/WORKFLOW_TRAINING.md`
- **Project Structure**: `.cursor/guides/project-structure.mdc`
- **Code Standards**: `.cursor/rules/coding-preferences.mdc`
