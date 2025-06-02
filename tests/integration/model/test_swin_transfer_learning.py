from typing import Any

import torch

from src.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
    SwinTransformerEncoderConfig,
)

EPSILON = 1e-6


def test_swin_transformer_encoder_freeze_layers_bool():
    """Test the freezing of layers using boolean flags."""
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers=True,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    trainable, frozen = count_trainable_params(encoder)
    assert trainable > 0, "No trainable parameters found"
    assert frozen > 0, "No frozen parameters found"
    assert frozen > trainable, "Expected majority of parameters to be frozen"


def test_swin_transformer_encoder_freeze_layers_str():
    """Test the freezing of layers using string patterns."""
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="patch_embed",
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    param_status = get_param_status_by_name(encoder)
    for name, is_trainable in param_status.items():
        if "patch_embed" in name:
            assert not is_trainable, f"Expected {name} to be frozen"

    config_all = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="all",
    )
    encoder_all_frozen = SwinTransformerEncoder(
        in_channels=3,
        config=config_all,
    )
    trainable, frozen = count_trainable_params(encoder_all_frozen)
    assert trainable == 0, "Expected all parameters to be frozen"
    assert frozen > 0, "No frozen parameters found"


def test_swin_transformer_encoder_freeze_layers_list():
    """Test the freezing of layers using a list of patterns."""
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers=["patch_embed", "stages.0"],
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    param_status = get_param_status_by_name(encoder)
    for name, is_trainable in param_status.items():
        if any(pattern in name for pattern in ["patch_embed", "stages.0"]):
            assert not is_trainable, f"Expected {name} to be frozen"


def test_swin_transformer_encoder_optimizer_param_groups():
    """
    Test the creation of optimizer parameter groups with differential LRs.
    """
    lr_scales = {"patch_embed": 0.1, "stages.0": 0.5, "stages.1": 0.8}
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        finetune_lr_scale=lr_scales,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    base_lr = 0.001
    param_groups = encoder.get_optimizer_param_groups(base_lr)
    assert len(param_groups) >= len(lr_scales), "Missing parameter groups"
    lr_by_group = {}
    for group in param_groups:
        if "name" in group:
            lr_by_group[group["name"]] = group["lr"]
    for pattern, scale in lr_scales.items():
        if pattern in lr_by_group:
            expected_lr = base_lr * scale
            assert (
                abs(lr_by_group[pattern] - expected_lr) < EPSILON
            ), f"Incorrect learning rate for {pattern}"


def test_swin_transformer_encoder_gradual_unfreeze():
    """Test the gradual unfreezing functionality."""
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="all",
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    trainable_before, _ = count_trainable_params(encoder)
    assert (
        trainable_before == 0
    ), "Expected all parameters to be frozen initially"
    print("\n=== MODEL STRUCTURE DEBUGGING ===")
    param_names = [name for name, _ in encoder.swin.named_parameters()]
    print(f"Total parameters: {len(param_names)}")
    prefixes = set()
    for name in param_names:
        parts = name.split(".")
        if len(parts) > 0:
            prefixes.add(parts[0])
    print(f"Top-level parameter groups: {sorted(prefixes)}")
    print("\nExamples of parameter names:")
    for i, name in enumerate(sorted(param_names)[:10]):
        print(f"  {i + 1}. {name}")
    unfreeze_schedule = {5: ["stages.0"], 10: ["stages.1"]}
    encoder.gradual_unfreeze(1, unfreeze_schedule)
    trainable, _ = count_trainable_params(encoder)
    assert trainable == 0, "No parameters should be unfrozen at epoch 1"
    print("\n=== APPLYING UNFREEZE AT EPOCH 5 ===")
    encoder.gradual_unfreeze(5, unfreeze_schedule)
    param_status = get_param_status_by_name(encoder)
    unfrozen_params = [
        name for name, is_trainable in param_status.items() if is_trainable
    ]
    print(f"\nUnfrozen parameters after epoch 5: {len(unfrozen_params)}")
    for i, name in enumerate(sorted(unfrozen_params)[:5]):
        print(f"  {i + 1}. {name}")
    layers0_unfrozen = False
    for name, is_trainable in param_status.items():
        if "layers_0" in name and is_trainable:
            layers0_unfrozen = True
            break
    assert (
        layers0_unfrozen
    ), "Expected layers_0 (equivalent to stages.0) to be unfrozen at epoch 5"
    encoder.gradual_unfreeze(10, unfreeze_schedule)
    param_status = get_param_status_by_name(encoder)
    layers1_unfrozen = False
    for name, is_trainable in param_status.items():
        if "layers_1" in name and is_trainable:
            layers1_unfrozen = True
            break
    assert (
        layers1_unfrozen
    ), "Expected layers_1 (equivalent to stages.1) to be unfrozen at epoch 10"


def test_integration_freeze_and_transfer():
    """
    Integration test for transfer learning with frozen layers and training.
    """
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="patch_embed",
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )
    x = torch.randn(2, 3, 256, 256)
    bottleneck, skip_connections = encoder(x)
    loss = bottleneck.mean() + sum(skip.mean() for skip in skip_connections)
    loss.backward()
    for name, param in encoder.named_parameters():
        if "patch_embed" in name:
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Expected no gradient for frozen parameter {name}"
        else:
            pass


def count_trainable_params(model: Any):
    """Count trainable and frozen parameters in a model."""
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    return trainable_params, frozen_params


def get_param_status_by_name(model: Any) -> dict[str, bool]:
    """Get a dictionary mapping parameter names to trainable status."""
    return {
        name: param.requires_grad for name, param in model.named_parameters()
    }
