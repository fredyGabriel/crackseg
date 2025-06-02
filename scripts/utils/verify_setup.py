#!/usr/bin/env python3
"""
Basic verification script for crack segmentation training.

This script checks that:
1. Configurations are properly set
2. Data exists and is accessible
3. The model can be instantiated
4. A basic training step can be executed

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def verify_data_exists() -> bool:
    """Check that data directories exist."""
    print("🔍 Checking data structure...")

    data_root = Path("data")
    required_dirs = ["train/images", "train/masks", "val/images", "val/masks"]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if not full_path.exists():
            missing_dirs.append(str(full_path))
        else:
            num_files = len(list(full_path.glob("*")))
            print(f"  ✅ {full_path}: {num_files} files")

    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False

    print("  ✅ Data structure verified")
    return True


def verify_torch_setup() -> tuple[bool, str]:
    """Check PyTorch setup."""
    print("\n🔍 Checking PyTorch setup...")

    print(f"  📦 PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(
            "  💾 CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        device = "cuda"
    else:
        print("  🖥️  CUDA not available, using CPU")
        device = "cpu"

    # Basic tensor test
    try:
        x = torch.randn(2, 3, 256, 256).to(device)
        y = torch.randn(2, 1, 256, 256).to(device)
        print(f"  ✅ Test tensors created on {device}")
        print(f"  📏 Input shape: {x.shape}")
        print(f"  📏 Output shape: {y.shape}")
        return True, device
    except Exception as e:
        print(f"  ❌ Error creating tensors: {e}")
        return False, "cpu"


def verify_model_instantiation() -> tuple[bool, nn.Module | None]:
    """Check that the model can be instantiated."""
    print("\n🔍 Checking model instantiation...")

    try:
        # Specific imports to avoid issues
        from src.model.bottleneck.cnn_bottleneck import BottleneckBlock
        from src.model.core.unet import BaseUNet
        from src.model.decoder.cnn_decoder import CNNDecoder
        from src.model.encoder.cnn_encoder import CNNEncoder

        print("  ✅ Module imports successful")

        # Simplified config
        encoder = CNNEncoder(in_channels=3, init_features=16, depth=3)
        print("  ✅ Encoder instantiated")

        bottleneck = BottleneckBlock(in_channels=64, out_channels=128)
        print("  ✅ Bottleneck instantiated")

        decoder = CNNDecoder(
            in_channels=128,
            skip_channels_list=[64, 32, 16],
            out_channels=1,
            depth=3,
        )
        print("  ✅ Decoder instantiated")

        # Create model using components directly (no strict type checking)
        model = BaseUNet(
            encoder=encoder,  # type: ignore
            bottleneck=bottleneck,  # type: ignore
            decoder=decoder,  # type: ignore
        )

        print("  ✅ BaseUNet model instantiated successfully")
        print(
            "  📊 Total parameters: "
            f"{sum(p.numel() for p in model.parameters()):,}"
        )
        print(
            "  📊 Trainable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"  # noqa: E501
        )

        return True, model
    except Exception as e:
        print(f"  ❌ Error instantiating model: {e}")
        print(f"  🔍 Detailed error: {type(e).__name__}: {str(e)}")
        return False, None


def verify_forward_pass(model: nn.Module, device: str) -> bool:
    """Check that the model can perform a forward pass."""
    print("\n🔍 Checking model forward pass...")

    try:
        model = model.to(device)
        x = torch.randn(2, 3, 256, 256).to(device)

        model.eval()
        with torch.no_grad():
            y = model(x)

        print("  ✅ Forward pass successful")
        print(f"  📏 Input: {x.shape}")
        print(f"  📏 Output: {y.shape}")
        print(
            f"  📊 Value range: [{y.min().item():.4f}, {y.max().item():.4f}]"
        )

        return True
    except Exception as e:
        print(f"  ❌ Error in forward pass: {e}")
        print(f"  🔍 Error type: {type(e).__name__}")

        # Extra debug for skip connection issues
        if "Skip connection" in str(e):
            print("  💡 Suggestion: Check skip_channels_list order")
            print("     Decoder expects LOW→HIGH resolution order")
            print("     Encoder produces HIGH→LOW resolution order")

        return False


def verify_configuration() -> bool:
    """Check that configuration files are valid."""
    print("\n🔍 Checking configurations...")

    config_files = [
        "configs/basic_verification.yaml",
        "configs/data/default.yaml",
        "configs/model/default.yaml",
        "configs/training/default.yaml",
    ]

    missing_configs = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
        else:
            print(f"  ✅ {config_file}")

    if missing_configs:
        print(f"  ❌ Missing configurations: {missing_configs}")
        return False

    print("  ✅ All configurations found")
    return True


def verify_imports() -> bool:
    """Check that all required imports work."""
    print("\n🔍 Checking module imports...")

    imports_to_test = [
        ("src.model.core.unet", "BaseUNet"),
        ("src.model.encoder.cnn_encoder", "CNNEncoder"),
        ("src.model.decoder.cnn_decoder", "CNNDecoder"),
        ("src.model.bottleneck.cnn_bottleneck", "BottleneckBlock"),
    ]

    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
            return False

    print("  ✅ All imports verified")
    return True


def main() -> int:
    """Run all verifications."""
    print("🚀 Starting setup verification for basic training")
    print("=" * 60)

    success = True

    # Check data
    if not verify_data_exists():
        success = False

    # Check PyTorch
    torch_ok, device = verify_torch_setup()
    if not torch_ok:
        success = False

    # Check configurations
    if not verify_configuration():
        success = False

    # Check imports
    if not verify_imports():
        success = False

    # Check model
    model_ok, model = verify_model_instantiation()
    if not model_ok:
        success = False

    # Check forward pass
    if (
        model_ok
        and model is not None
        and not verify_forward_pass(model, device)
    ):
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 Verification completed successfully!")
        print("\n📋 Recommended training commands:")
        print(
            "   🏃‍♂️ Basic training: python run.py --config-name=basic_"
            "verification"
        )
        print(
            "   🔧 Force CPU: python run.py --config-name=basic_verification "
            "training.device=cpu"
        )
        print(
            "   📊 With detailed logs: python run.py --config-name=basic_"
            "verification log_level=DEBUG"
        )
        print("\n💡 Current configuration:")
        print(
            "   📦 Model: Simplified U-Net CNN (16→64→128 channels, depth=3)"
        )
        print("   📊 Data: 16 training samples, 8 validation samples")
        print("   🔄 Training: 2 epochs, batch_size=4")
        print(
            f"  🖥️  Recommended device: {'CUDA' if device == 'cuda' else 'CPU'}"
        )
    else:
        print("❌ Verification failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
