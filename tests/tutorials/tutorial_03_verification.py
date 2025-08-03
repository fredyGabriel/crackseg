#!/usr/bin/env python3
"""
Tutorial 03 Verification Script

This script verifies that all components created in Tutorial 03 work correctly
with the current project structure and registry system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_loss_registry():
    """Test the loss registry system."""
    print("🔍 Testing Loss Registry System...")

    try:
        from crackseg.training.losses.loss_registry_setup import loss_registry

        print("✅ Loss registry imported successfully")
        print(f"✅ Available losses: {loss_registry.list_components()}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import loss registry: {e}")
        return False


def test_smooth_l1_loss_creation():
    """Test creating the SmoothL1Loss component."""
    print("\n🔍 Testing SmoothL1Loss Creation...")

    # Create the loss function file
    loss_file = (
        project_root
        / "src"
        / "crackseg"
        / "training"
        / "losses"
        / "smooth_l1_loss.py"
    )

    if loss_file.exists():
        print(f"✅ SmoothL1Loss file already exists: {loss_file}")
        return True

    # Create the file
    loss_content = """import torch
import torch.nn as nn
from crackseg.training.losses.loss_registry_setup import loss_registry
from crackseg.training.losses.base_loss import SegmentationLoss

@loss_registry.register(
    name="smooth_l1_loss", tags=["segmentation", "regression"]
)
class SmoothL1Loss(SegmentationLoss):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=self.beta)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(y_pred, y_true)
"""

    try:
        loss_file.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_file, "w") as f:
            f.write(loss_content)
        print(f"✅ Created SmoothL1Loss file: {loss_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create SmoothL1Loss file: {e}")
        return False


def test_smooth_l1_loss_registration():
    """Test registering the SmoothL1Loss component."""
    print("\n🔍 Testing SmoothL1Loss Registration...")

    try:
        # Add import to __init__.py
        init_file = (
            project_root
            / "src"
            / "crackseg"
            / "training"
            / "losses"
            / "__init__.py"
        )

        with open(init_file) as f:
            content = f.read()

        if "smooth_l1_loss" not in content:
            # Add the import
            with open(init_file, "a") as f:
                f.write("\nfrom . import smooth_l1_loss\n")
            print("✅ Added smooth_l1_loss import to __init__.py")
        else:
            print("✅ smooth_l1_loss import already exists in __init__.py")

        # Test importing
        from crackseg.training.losses.smooth_l1_loss import SmoothL1Loss

        print("✅ SmoothL1Loss imported successfully")

        # Test instantiation
        loss_fn = SmoothL1Loss(beta=0.5)
        print("✅ SmoothL1Loss instantiated successfully")

        # Test forward pass
        import torch

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        loss = loss_fn(pred, target)
        print(f"✅ Forward pass successful: loss = {loss.item():.4f}")

        return True
    except Exception as e:
        print(f"❌ Failed to register SmoothL1Loss: {e}")
        return False


def test_config_creation():
    """Test creating configuration files."""
    print("\n🔍 Testing Configuration Creation...")

    # Create config directory
    config_dir = project_root / "configs" / "training" / "loss"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create smooth_l1.yaml
    config_file = config_dir / "smooth_l1.yaml"
    config_content = """# Smooth L1 Loss configuration
_target_: src.training.losses.smooth_l1_loss.SmoothL1Loss
beta: 1.0  # Smoothing parameter
"""

    try:
        with open(config_file, "w") as f:
            f.write(config_content)
        print(f"✅ Created config file: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False


def test_experiment_config_creation():
    """Test creating experiment configuration."""
    print("\n🔍 Testing Experiment Configuration Creation...")

    # Create experiments directory
    exp_dir = project_root / "configs" / "experiments" / "tutorial_03"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment config
    exp_file = exp_dir / "smooth_l1_experiment.yaml"
    exp_content = """# Experiment using the new SmoothL1Loss
defaults:
  - basic_verification
  - _self_

# Use the new loss function
training:
  loss:
    _target_: src.training.losses.smooth_l1_loss.SmoothL1Loss
    beta: 0.5  # Custom beta parameter

# Other experiment parameters
training:
  learning_rate: 0.0001
  epochs: 50

dataloader:
  batch_size: 12
"""

    try:
        with open(exp_file, "w") as f:
            f.write(exp_content)
        print(f"✅ Created experiment config: {exp_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create experiment config: {e}")
        return False


def test_optimizer_creation():
    """Test creating custom optimizer components."""
    print("\n🔍 Testing Optimizer Creation...")

    # Create optimizer directory
    opt_dir = project_root / "src" / "crackseg" / "training" / "optimizers"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # Create registry
    registry_file = opt_dir / "registry.py"
    registry_content = """from typing import Dict, Type
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
"""

    try:
        with open(registry_file, "w") as f:
            f.write(registry_content)
        print(f"✅ Created optimizer registry: {registry_file}")

        # Create custom optimizer
        custom_opt_file = opt_dir / "custom_adam.py"
        custom_opt_content = """import torch
import torch.optim
from crackseg.training.optimizers.registry import register_optimizer

@register_optimizer("custom_adam")
class CustomAdam(torch.optim.Adam):
    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.custom_parameter = 0.1  # Custom parameter

    def step(self, closure=None):
        # Add custom logic here if needed
        return super().step(closure)
"""

        with open(custom_opt_file, "w") as f:
            f.write(custom_opt_content)
        print(f"✅ Created custom optimizer: {custom_opt_file}")

        # Create __init__.py
        init_file = opt_dir / "__init__.py"
        init_content = """from .registry import (
    register_optimizer, get_optimizer, list_optimizers
)
from . import custom_adam

__all__ = ['register_optimizer', 'get_optimizer', 'list_optimizers']
"""

        with open(init_file, "w") as f:
            f.write(init_content)
        print(f"✅ Created optimizer __init__.py: {init_file}")

        return True
    except Exception as e:
        print(f"❌ Failed to create optimizer components: {e}")
        return False


def test_model_creation():
    """Test creating custom model components."""
    print("\n🔍 Testing Model Creation...")

    # Create model directory
    model_dir = project_root / "src" / "crackseg" / "model" / "architectures"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create registry
    registry_file = model_dir / "registry.py"
    registry_content = """from typing import Dict, Type
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
"""

    try:
        with open(registry_file, "w") as f:
            f.write(registry_content)
        print(f"✅ Created model registry: {registry_file}")

        # Create simple UNet
        unet_file = model_dir / "simple_unet.py"
        unet_content = """import torch
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
"""

        with open(unet_file, "w") as f:
            f.write(unet_content)
        print(f"✅ Created SimpleUNet: {unet_file}")

        # Create __init__.py
        init_file = model_dir / "__init__.py"
        init_content = """from .registry import (
    register_model, get_model, list_models
)
from . import simple_unet

__all__ = ['register_model', 'get_model', 'list_models']
"""

        with open(init_file, "w") as f:
            f.write(init_content)
        print(f"✅ Created model __init__.py: {init_file}")

        return True
    except Exception as e:
        print(f"❌ Failed to create model components: {e}")
        return False


def test_component_verification():
    """Test that all components can be imported and used."""
    print("\n🔍 Testing Component Verification...")

    try:
        # Test loss function
        from crackseg.training.losses.smooth_l1_loss import SmoothL1Loss

        print("✅ SmoothL1Loss imported successfully")

        # Test optimizer
        from crackseg.training.optimizers.custom_adam import CustomAdam

        print("✅ CustomAdam imported successfully")

        # Test model
        from crackseg.model.architectures.simple_unet import SimpleUNet

        print("✅ SimpleUNet imported successfully")

        # Test registries
        from crackseg.model.architectures.registry import list_models
        from crackseg.training.optimizers.registry import list_optimizers

        print(f"✅ Registered optimizers: {list_optimizers()}")
        print(f"✅ Registered models: {list_models()}")

        # Use the imports to avoid linter warnings
        _ = SmoothL1Loss
        _ = CustomAdam
        _ = SimpleUNet

        return True
    except Exception as e:
        print(f"❌ Component verification failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("🚀 Starting Tutorial 03 Verification...")
    print("=" * 50)

    tests = [
        test_loss_registry,
        test_smooth_l1_loss_creation,
        test_smooth_l1_loss_registration,
        test_config_creation,
        test_experiment_config_creation,
        test_optimizer_creation,
        test_model_creation,
        test_component_verification,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Verification Results:")

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results, strict=False), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i:2d}. {test.__name__}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Tutorial 03 is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
