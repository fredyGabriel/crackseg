# Tutorial 3: Extending the Project with Custom Components (CLI Only)

This advanced tutorial walks you through the process of adding new, custom components to the project
using the command line interface (CLI) only. We will create a new loss function and make it
available to the configuration system.

**Goal**: Understand the factory and registry pattern to add custom code.
**Prerequisite**: [Tutorial 2: Creating Custom Experiments (CLI)](02_custom_experiment_cli.md)

## Prerequisites

- You have completed Tutorial 2 and understand custom experiment configuration.
- **CRITICAL**: You have installed the `crackseg` package (`pip install -e . --no-deps`).
- You have verified the installation works (`python -c "import crackseg"`).
- You understand that the package must be installed for imports to work.

## Step 1: Understand the Registry System

First, let's examine how the registry system works:

```bash
conda activate crackseg
# Look at the loss registry
cat src/crackseg/training/losses/registry.py

# Look at existing loss functions
dir src/crackseg/training/losses/

# Check the __init__.py file
cat src/crackseg/training/losses/__init__.py
```

The registry system uses decorators to automatically register components when modules are imported.

## Step 2: Create a New Loss Function

Let's create a new loss function called "Smooth L1 Loss":

```bash
conda activate crackseg
# Create the new loss function file
cat > src/crackseg/training/losses/smooth_l1_loss.py << 'EOF'
import torch
import torch.nn as nn
from crackseg.training.losses.registry import register_loss

@register_loss("smooth_l1")
class SmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=self.beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Your custom logic here, if any.
        # For this example, we directly use PyTorch's implementation.
        return self.loss_fn(y_pred, y_true)
EOF
```

**Important**: The import `from crackseg.training.losses.registry import register_loss`
only works if the `crackseg` package is installed. If you get import errors, run:

```bash
conda activate crackseg
pip install -e . --no-deps
```

## Step 3: Register the Component

The key to making your component available to the system is the `@register_loss`
decorator, which we imported from `crackseg.training.losses.registry`.

```python
@register_loss("smooth_l1")
```

- `@register_loss(...)`: This function adds your class to a central dictionary (the "registry").
- `"smooth_l1"`: This is the **unique name** you will use in the configuration files
    to refer to your new loss function.

## Step 4: Make the Component Discoverable

The registry system needs to import your file to find the decorated class.

```bash
conda activate crackseg
# Add import to __init__.py
echo "from . import smooth_l1_loss" >> src/crackseg/training/losses/__init__.py

# Verify the import was added
Get-Content src/crackseg/training/losses/__init__.py | Select-Object -Last 5
```

By importing the module in the package's `__init__.py`, you ensure that the
`@register_loss` decorator runs when the `losses` package is loaded, making your
new loss available to the factory.

## Step 5: Create the Configuration File

Now, let's create the corresponding YAML file so Hydra can use your new loss:

```bash
conda activate crackseg
# Create the configuration file
cat > configs/training/loss/smooth_l1.yaml << 'EOF'
_target_: crackseg.training.losses.smooth_l1_loss.SmoothL1Loss
_name_: smooth_l1

# You can define default parameters here
beta: 1.0
EOF
```

- `_target_`: This is the **full Python path** to your new class (note: uses `crackseg.` prefix).
- `_name_`: This should match the **unique name** you used in the decorator.
- `beta: 1.0`: You can expose parameters to be configured via Hydra.

## Step 6: Test Your Component

Before using it in training, let's test that your component works correctly:

```bash
conda activate crackseg
# Test the component directly
python -c "
import torch
from crackseg.training.losses.smooth_l1_loss import SmoothL1Loss

# Create test data
pred = torch.randn(2, 1, 64, 64)
target = torch.randn(2, 1, 64, 64)

# Test the loss function
loss_fn = SmoothL1Loss(beta=0.5)
loss = loss_fn(pred, target)

print(f'✅ SmoothL1Loss works: {loss.item():.4f}')
print(f'✅ Loss shape: {loss.shape}')
print(f'✅ Component registered successfully')
"
```

## Step 7: Use Your New Component

You can now use your new loss function in any experiment by overriding the
configuration from the command line:

```bash
conda activate crackseg
# Use your new loss function
python run.py --config-name basic_verification training.loss=smooth_l1 training.loss.beta=0.5
```

- `training.loss=smooth_l1`: Selects your new loss.
- `training.loss.beta=0.5`: Overrides the `beta` parameter within your loss config.

## Step 8: Create a Custom Experiment with Your Component

Create a new experiment configuration that uses your component:

```bash
conda activate crackseg
# Create experiment configuration
cat > generated_configs/smooth_l1_experiment.yaml << 'EOF'
# Experiment using the new SmoothL1Loss
defaults:
  - basic_verification
  - _self_

# Use the new loss function
training:
  loss:
    _target_: crackseg.training.losses.smooth_l1_loss.SmoothL1Loss
    beta: 0.5  # Custom beta parameter

# Other experiment parameters
training:
  learning_rate: 0.0001
  epochs: 50

dataloader:
  batch_size: 12
EOF
```

## Step 9: Run the Experiment

**IMPORTANT**: Use `run.py` instead of `src/main.py` for proper execution:

```bash
conda activate crackseg
# Run the experiment
python run.py --config-name smooth_l1_experiment
```

## Step 10: Create Additional Custom Components

Let's create another custom component - a new optimizer:

### Create a Custom Optimizer

```bash
conda activate crackseg
# Create optimizer registry if it doesn't exist
mkdir -p src/crackseg/training/optimizers

# Create the optimizer registry
cat > src/crackseg/training/optimizers/registry.py << 'EOF'
from typing import Dict, Type
import torch.optim

_OPTIMIZER_REGISTRY: Dict[str, Type[torch.optim.Optimizer]] = {}

def register_optimizer(name: str):
    def decorator(cls):
        _OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

def get_optimizer(name: str) -> Type[torch.optim.Optimizer]:
    return _OPTIMIZER_REGISTRY[name]

def list_optimizers() -> list:
    return list(_OPTIMIZER_REGISTRY.keys())
EOF

# Create a custom optimizer
cat > src/crackseg/training/optimizers/custom_adam.py << 'EOF'
import torch
import torch.optim
from crackseg.training.optimizers.registry import register_optimizer

@register_optimizer("custom_adam")
class CustomAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.custom_parameter = 0.1  # Custom parameter

    def step(self, closure=None):
        # Add custom logic here if needed
        return super().step(closure)
EOF

# Create __init__.py for optimizers
cat > src/crackseg/training/optimizers/__init__.py << 'EOF'
from .registry import register_optimizer, get_optimizer, list_optimizers
from . import custom_adam

__all__ = ['register_optimizer', 'get_optimizer', 'list_optimizers']
EOF

# Add optimizers to training __init__.py
echo "from . import optimizers" >> src/crackseg/training/__init__.py
```

### Create Optimizer Configuration

```bash
conda activate crackseg
# Create optimizer config directory
mkdir -p configs/training/optimizer

# Create configuration file
cat > configs/training/optimizer/custom_adam.yaml << 'EOF'
_target_: crackseg.training.optimizers.custom_adam.CustomAdam
_name_: custom_adam

# Default parameters
lr: 0.001
betas: [0.9, 0.999]
eps: 1e-8
weight_decay: 0.0001
EOF
```

### Test the New Optimizer

```bash
conda activate crackseg
# Test the optimizer
python -c "
import torch
from crackseg.training.optimizers.custom_adam import CustomAdam

# Create test parameters
params = [torch.randn(10, 10, requires_grad=True)]

# Test the optimizer
optimizer = CustomAdam(params, lr=0.001)
print(f'✅ CustomAdam created successfully')
print(f'✅ Custom parameter: {optimizer.custom_parameter}')
"
```

## Step 11: Create a Model Component

Let's create a custom model component:

### Create a Custom Model

```bash
conda activate crackseg
# Create model registry if it doesn't exist
mkdir -p src/crackseg/model/architectures

# Create the model registry
cat > src/crackseg/model/architectures/registry.py << 'EOF'
from typing import Dict, Type
import torch.nn

_MODEL_REGISTRY: Dict[str, Type[torch.nn.Module]] = {}

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str) -> Type[torch.nn.Module]:
    return _MODEL_REGISTRY[name]

def list_models() -> list:
    return list(_MODEL_REGISTRY.keys())
EOF

# Create a simple custom model
cat > src/crackseg/model/architectures/simple_unet.py << 'EOF'
import torch
import torch.nn as nn
from crackseg.model.architectures.registry import register_model

@register_model("simple_unet")
class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)
EOF

# Create __init__.py for architectures
cat > src/crackseg/model/architectures/__init__.py << 'EOF'
from .registry import register_model, get_model, list_models
from . import simple_unet

__all__ = ['register_model', 'get_model', 'list_models']
EOF

# Add architectures to model __init__.py
echo "from . import architectures" >> src/crackseg/model/__init__.py
```

### Create Model Configuration

```bash
conda activate crackseg
# Create model config directory
mkdir -p configs/model/architecture

# Create configuration file
cat > configs/model/architecture/simple_unet.yaml << 'EOF'
_target_: crackseg.model.architectures.simple_unet.SimpleUNet
_name_: simple_unet

# Model parameters
in_channels: 3
out_channels: 1
EOF
```

### Test the New Model

```bash
conda activate crackseg
# Test the model
python -c "
import torch
from crackseg.model.architectures.simple_unet import SimpleUNet

# Create test input
x = torch.randn(1, 3, 256, 256)

# Test the model
model = SimpleUNet(in_channels=3, out_channels=1)
output = model(x)

print(f'✅ SimpleUNet created successfully')
print(f'✅ Input shape: {x.shape}')
print(f'✅ Output shape: {output.shape}')
print(f'✅ Output range: [{output.min():.4f}, {output.max():.4f}]')
"
```

## Step 12: Create a Complete Custom Experiment

Now let's create an experiment that uses all your custom components:

```bash
conda activate crackseg
# Create comprehensive experiment
cat > generated_configs/custom_components_experiment.yaml << 'EOF'
# Experiment using all custom components
defaults:
  - basic_verification
  - _self_

# Use custom model
model:
  _target_: crackseg.model.architectures.simple_unet.SimpleUNet
  in_channels: 3
  out_channels: 1

# Use custom loss
training:
  loss:
    _target_: crackseg.training.losses.smooth_l1_loss.SmoothL1Loss
    beta: 0.5

# Use custom optimizer
training:
  optimizer:
    _target_: crackseg.training.optimizers.custom_adam.CustomAdam
    lr: 0.001
    weight_decay: 0.0001

# Training parameters
training:
  epochs: 30
  learning_rate: 0.001

dataloader:
  batch_size: 8
EOF
```

## Step 13: Run the Complete Experiment

**IMPORTANT**: Use `run.py` instead of `src/main.py` for proper execution:

```bash
conda activate crackseg
# Run the experiment with all custom components
python run.py --config-name custom_components_experiment
```

## Step 14: Verify Component Registration

Check that all your components are properly registered:

```bash
conda activate crackseg
# Verify loss functions
python -c "
from crackseg.training.losses.registry import list_losses
print('Registered losses:', list_losses())
"

# Verify optimizers
python -c "
from crackseg.training.optimizers.registry import list_optimizers
print('Registered optimizers:', list_optimizers())
"

# Verify models
python -c "
from crackseg.model.architectures.registry import list_models
print('Registered models:', list_models())
"
```

## Step 15: Quality Assurance

After adding new components, run quality gates:

```bash
conda activate crackseg
# Format code
black .

# Lint code
python -m ruff . --fix

# Type check
basedpyright .

# Run tests (if available)
python -m pytest tests/unit/training/test_losses.py -v
```

## Troubleshooting

### Common Issues

1. **Import Error**: No module named 'crackseg'

    - Solution: Run `pip install -e . --no-deps` from the project root

2. **Import Error**: No module named 'crackseg.training.losses.registry'

    - Solution: Ensure the package is installed and the registry file exists

3. **Configuration Error**: Component not found

    - Solution: Check that the component is properly registered in `__init__.py`
    - Verify the `_target_` path in the YAML file

4. **Hydra Instantiation Error**

    - Solution: Check YAML syntax and ensure all required parameters are provided

5. **Registry Not Working**

    - Solution: Ensure the module is imported in `__init__.py`
    - Restart Python session after adding new components

### Debugging Component Registration

```bash
conda activate crackseg
# Debug registry contents
python -c "
import crackseg.training.losses
from crackseg.training.losses.registry import list_losses
print('Available losses:', list_losses())

import crackseg.training.optimizers
from crackseg.training.optimizers.registry import list_optimizers
print('Available optimizers:', list_optimizers())

import crackseg.model.architectures
from crackseg.model.architectures.registry import list_models
print('Available models:', list_models())
"
```

## Summary

This modular, config-driven approach allows for rapid prototyping and testing of
new components and ideas. The registry pattern makes it easy to add new components
without modifying existing code.

## Quick Reference Commands

```bash
# Create new component
conda activate crackseg
cat > src/crackseg/training/losses/my_loss.py << 'EOF'
import torch.nn as nn
from crackseg.training.losses.registry import register_loss

@register_loss("my_loss")
class MyLoss(nn.Module):
    def __init__(self, param=1.0):
        super().__init__()
        self.param = param

    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target) * self.param
EOF

# Register component
echo "from . import my_loss" >> src/crackseg/training/losses/__init__.py

# Create config
cat > configs/training/loss/my_loss.yaml << 'EOF'
_target_: crackseg.training.losses.my_loss.MyLoss
_name_: my_loss
param: 1.0
EOF

# Test component
conda activate crackseg
python -c "from crackseg.training.losses.my_loss import MyLoss; print('✅ Component works')"

# Use in experiment
conda activate crackseg
python run.py --config-name basic_verification training.loss=my_loss
```

## Summary of Corrections Made

1. **Entry Point**: Changed from `src/main.py` to `run.py` for proper execution
2. **Base Configuration**: Use `basic_verification` instead of `base` to avoid Hydra dependency issues
3. **PowerShell Commands**: Updated `ls` to `dir` and `tail` to `Get-Content | Select-Object`
4. **Configuration Structure**: Updated to use proper `_target_` syntax for component instantiation
5. **Component Registration**: Ensured proper import paths and registry usage

---

**Congratulations!** You have successfully extended the project with multiple custom components
using the CLI approach. This same pattern applies to adding new data transforms, evaluation metrics,
and more.
