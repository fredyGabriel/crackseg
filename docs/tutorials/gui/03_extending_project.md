# Tutorial 3: Extending the Project with a New Component

**TODO: This tutorial needs verification and testing with the current GUI implementation.**

This advanced tutorial walks you through the process of adding a new, custom component
to the project. We will create a new loss function and make it available to the
configuration system.

**Goal**: Understand the factory and registry pattern to add custom code.
**Prerequisite**: [Tutorial 2: Configuring a New Experiment](02_custom_experiment.md)

## Prerequisites

- You have completed Tutorial 2 and understand custom experiment configuration.
- **CRITICAL**: You have installed the `crackseg` package (`pip install -e . --no-deps`).
- You have verified the installation works (`python -c "import crackseg"`).
- You understand that the package must be installed for imports to work.

---

## Step 1: Create the New Component File

Let's create a new loss function called "Smooth L1 Loss".

1. Navigate to the `src/training/losses/` directory.
2. Create a new Python file named `smooth_l1_loss.py`.
3. Inside this file, define your loss function. It must be a class that inherits
    from `torch.nn.Module`.

```python
# src/training/losses/smooth_l1_loss.py
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
```

**Important**: The import `from crackseg.training.losses.registry import register_loss`
only works if the `crackseg` package is installed. If you get import errors, run:

```bash
conda activate crackseg
pip install -e . --no-deps
```

## Step 2: Register the Component

The key to making your component available to the system is the `@register_loss`
decorator, which we imported from `src.training.losses.registry`.

```python
@register_loss("smooth_l1")
```

- `@register_loss(...)`: This function adds your class to a central dictionary (the "registry").
- `"smooth_l1"`: This is the **unique name** you will use in the configuration files
    to refer to your new loss function.

## Step 3: Make the Component Discoverable

The registry system needs to import your file to find the decorated class.

1. Open `src/training/losses/__init__.py`.
2. Add a line to import your new module:

```python
# src/training/losses/__init__.py

# ... other imports
from . import smooth_l1_loss
```

By importing the module in the package's `__init__.py`, you ensure that the
`@register_loss` decorator runs when the `losses` package is loaded, making your
new loss available to the factory.

## Step 4: Create the Configuration File

Now, let's create the corresponding YAML file so Hydra can use your new loss.

1. Navigate to `configs/training/loss/`.
2. Create a new file named `smooth_l1.yaml`.
3. Add the following content:

```yaml
# configs/training/loss/smooth_l1.yaml
_target_: crackseg.training.losses.smooth_l1_loss.SmoothL1Loss
_name_: smooth_l1

# You can define default parameters here
beta: 1.0
```

- `_target_`: This is the **full Python path** to your new class (note: uses `crackseg.` prefix).
- `_name_`: This should match the **unique name** you used in the decorator.
- `beta: 1.0`: You can expose parameters to be configured via Hydra.

## Step 5: Use Your New Component

You can now use your new loss function in any experiment by overriding the
configuration from the command line:

```bash
conda activate crackseg
python run.py --config-name basic_verification training.loss=smooth_l1 training.loss.beta=0.5
```

- `training/loss=smooth_l1`: Selects your new loss.
- `training/loss.beta=0.5`: Overrides the `beta` parameter within your loss config.

### Alternative: Create and Use Your Experiment Config

Create a new main experiment file (e.g., `generated_configs/smooth_l1_exp.yaml`) or
modify an existing one using the GUI editor. To use your new loss, change
the defaults section:

```yaml
# In your main experiment config
defaults:
  - training/loss: smooth_l1
  # ... other defaults
```

Now, when you run this experiment, the training pipeline will automatically
instantiate `SmoothL1Loss` with the specified parameters.

## Step 6: Running the Experiment

### Option A: Using CLI

```bash
conda activate crackseg
python run.py --config-name smooth_l1_exp
```

### Option B: Using the GUI

1. Launch the GUI and navigate to the **Experiment Configuration** page.
2. Load the experiment configuration file that uses your new component (e.g.,
    `smooth_l1_exp.yaml`).
3. Set a **Run Directory**.
4. Navigate to the **Train** page by clicking **"🚀 Start Training"**.
5. Start the process and monitor the training. The system will now use your
    custom `SmoothL1Loss` component.

## Verification and Testing

### Test Your Component

Verify that your component works correctly:

```bash
conda activate crackseg
python -c "
import torch
from crackseg.training.losses.smooth_l1_loss import SmoothL1Loss
loss_fn = SmoothL1Loss(beta=0.5)
pred = torch.randn(2, 1, 64, 64)
target = torch.randn(2, 1, 64, 64)
loss = loss_fn(pred, target)
print(f'✅ SmoothL1Loss works: {loss.item():.4f}')
"
```

### Quality Gates

After adding new components, run quality gates:

```bash
conda activate crackseg
black .
python -m ruff . --fix
basedpyright .
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'crackseg'**

    - Solution: Run `pip install -e . --no-deps` from the project root

2. **Import Error: No module named 'crackseg.training.losses.registry'**

    - Solution: Ensure the package is installed and the registry file exists

3. **Configuration Error: Component not found**

    - Solution: Check that the component is properly registered in `__init__.py`
    - Verify the `_target_` path in the YAML file

4. **Hydra Instantiation Error**

    - Solution: Check YAML syntax and ensure all required parameters are provided

## Summary

This modular, config-driven approach allows for rapid prototyping and testing of
new components and ideas.

---

**Congratulations!** You have successfully extended the project with a new,
configurable component. This same pattern applies to adding new models, encoders,
decoders, data transforms, and more.
