"""
Transfer learning demonstration for the Swin Transformer V2 encoder.

This script demonstrates advanced transfer learning techniques for the
SwinTransformerEncoder, including:
- Parameter freezing strategies
- Gradual unfreezing
- Differential learning rates

These techniques help to efficiently fine-tune pre-trained models on new
datasets with limited data or computational resources.
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import after configuring the path
from crackseg.model.encoder.swin_transformer_encoder import (  # noqa: E402
    SwinTransformerEncoder,  # noqa: E402
    SwinTransformerEncoderConfig,  # noqa: E402
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(num_samples=100, img_size=256, in_channels=3):
    """Create a synthetic dataset for demonstration purposes."""
    # Create random input images
    X = torch.randn(num_samples, in_channels, img_size, img_size)

    # Create random target feature maps (simulating a task)
    Y = torch.randn(num_samples, 768, img_size // 32, img_size // 32)

    # Create dataloaders
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    return dataloader


def display_model_info(encoder, title="Model Information"):
    """Display information about the model's trainable parameters."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for _name, param in encoder.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    print(f"\n{title}")
    print(f"Total parameters: {total_params:,}")
    pct = trainable_params / total_params
    print(f"Trainable parameters: {trainable_params:,} ({pct:.1%})")
    pct = frozen_params / total_params
    print(f"Frozen parameters: {frozen_params:,} ({pct:.1%})")

    # Print out layer groups for learning rates if using differential LRs
    if hasattr(encoder, "finetune_lr_scale") and encoder.finetune_lr_scale:
        print("\nLearning Rate Scales:")
        for pattern, scale in encoder.finetune_lr_scale.items():
            print(f"  - {pattern}: {scale:.2f}x base learning rate")


def setup_encoder_with_freezing(freeze_mode):
    """Set up an encoder with different freezing strategies."""
    config = SwinTransformerEncoderConfig(
        freeze_layers=freeze_mode,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    return encoder


def setup_encoder_with_differential_lr():
    """Set up an encoder with differential learning rates for layers."""
    # Define learning rate scales for different layers
    lr_scales = {
        "patch_embed": 0.1,  # Early layers learn very slowly (10% of base LR)
        "stages.0": 0.3,  # First stage learns slowly (30% of base LR)
        "stages.1": 0.7,  # Middle stage learns faster (70% of base LR)
        "stages.2": 1.0,  # Final stage learns at full rate (100% of base LR)
    }
    config = SwinTransformerEncoderConfig(
        finetune_lr_scale=lr_scales,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    return encoder


def create_optimizer_with_param_groups(encoder, base_lr=0.001):
    """Create an AdamW optimizer with parameter groups from encoder."""
    param_groups = encoder.get_optimizer_param_groups(base_lr)
    optimizer = optim.AdamW(param_groups)

    return optimizer


def simulate_gradual_unfreeze_training(encoder, num_epochs=20):
    """Simulate a training process with gradual unfreezing."""
    # Define a schedule for when to unfreeze different layers
    unfreeze_schedule = {
        5: ["stages.2"],  # Unfreeze the last stage at epoch 5
        10: ["stages.1"],  # Unfreeze the middle stage at epoch 10
        15: ["stages.0"],  # Unfreeze the first stage at epoch 15
        18: ["patch_embed"],  # Unfreeze the patch embedding layer at epoch 18
    }

    # Print initial state
    print("\nSimulating gradual unfreezing over 20 epochs:")
    display_model_info(encoder, "Initial state (Epoch 0)")

    # Simulate training epochs
    for epoch in range(1, num_epochs + 1):
        # Update unfreezing based on current epoch
        encoder.gradual_unfreeze(epoch, unfreeze_schedule)

        # Every 5 epochs, show current parameter stats
        if epoch in [5, 10, 15, 20]:
            display_model_info(encoder, f"After Epoch {epoch}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Demonstrate various transfer learning techniques."""
    print("\n=== DEMO 1: Different Parameter Freezing Strategies ===")

    # 1. No freezing (full fine-tuning)
    encoder_full = setup_encoder_with_freezing(False)
    display_model_info(encoder_full, "1. Full Fine-tuning (No Freezing)")

    # 2. Freeze all but the last block (common strategy)
    encoder_partial = setup_encoder_with_freezing(True)
    display_model_info(
        encoder_partial, "2. Partial Fine-tuning (Last Stage Only)"
    )

    # 3. Freeze specific stages
    encoder_custom = setup_encoder_with_freezing("patch_embed,stages.0")
    display_model_info(
        encoder_custom, "3. Custom Freezing (Patch Embed + First Stage)"
    )

    # 4. Feature extraction (all frozen)
    encoder_frozen = setup_encoder_with_freezing("all")
    display_model_info(encoder_frozen, "4. Feature Extraction (All Frozen)")

    print("\n=== DEMO 2: Optimizer with Differential Learning Rates ===")
    encoder_diff_lr = setup_encoder_with_differential_lr()
    optimizer = create_optimizer_with_param_groups(
        encoder_diff_lr, base_lr=0.001
    )

    # Display the learning rates for each parameter group
    print("\nOptimizer Parameter Groups:")
    for i, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"Group {i}")
        print(f"  - {group_name}: lr = {group['lr']}")

    print("\n=== DEMO 3: Gradual Unfreezing During Training ===")
    # Start with everything frozen except the very last layer
    encoder_gradual = setup_encoder_with_freezing(
        "patch_embed,stages.0,stages.1,stages.2"
    )
    simulate_gradual_unfreeze_training(encoder_gradual)

    print("\nAll transfer learning techniques demonstrated successfully!")


if __name__ == "__main__":
    # Hydra will automatically provide the cfg parameter when called from
    # command line
    # For direct execution, we need to handle this differently
    import sys

    if len(sys.argv) > 1:
        # If arguments are provided, let Hydra handle them
        main()
    else:
        # For direct execution without arguments, we need to provide a default
        # config
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = compose(config_name="base")
            main(cfg)
