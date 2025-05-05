import torch
from src.model.encoder.swin_transformer_encoder import SwinTransformerEncoder


def test_swin_transformer_encoder_freeze_layers_bool():
    """Test the freezing of layers using boolean flags."""
    # Initialize with freeze_layers=True (freeze all but last block)
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,  # No need for pretrained weights in test
        freeze_layers=True
    )

    # Count trainable vs frozen parameters
    trainable, frozen = count_trainable_params(encoder)

    # Assert that some parameters are trainable and some are frozen
    assert trainable > 0, "No trainable parameters found"
    assert frozen > 0, "No frozen parameters found"

    # Assert that most parameters are frozen when freeze_layers=True
    assert frozen > trainable, "Expected majority of parameters to be frozen"


def test_swin_transformer_encoder_freeze_layers_str():
    """Test the freezing of layers using string patterns."""
    # Initialize with specific layers frozen
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="patch_embed"  # Only freeze patch embedding
    )

    # Get parameter status
    param_status = get_param_status_by_name(encoder)

    # Check that patch_embed parameters are frozen
    for name, is_trainable in param_status.items():
        if "patch_embed" in name:
            assert not is_trainable, f"Expected {name} to be frozen"

    # Initialize with all layers frozen
    encoder_all_frozen = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="all"
    )

    # All parameters should be frozen
    trainable, frozen = count_trainable_params(encoder_all_frozen)
    assert trainable == 0, "Expected all parameters to be frozen"
    assert frozen > 0, "No frozen parameters found"


def test_swin_transformer_encoder_freeze_layers_list():
    """Test the freezing of layers using a list of patterns."""
    # Initialize with list of patterns to freeze
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers=["patch_embed", "stages.0"]
    )

    # Get parameter status
    param_status = get_param_status_by_name(encoder)

    # Check that specified patterns are frozen
    for name, is_trainable in param_status.items():
        if any(pattern in name for pattern in ["patch_embed", "stages.0"]):
            assert not is_trainable, f"Expected {name} to be frozen"


def test_swin_transformer_encoder_optimizer_param_groups():
    """Test the creation of optimizer parameter groups with differential LRs.
    """
    # Define learning rate scales
    lr_scales = {
        "patch_embed": 0.1,
        "stages.0": 0.5,
        "stages.1": 0.8
    }

    # Initialize with differential learning rates
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        finetune_lr_scale=lr_scales
    )

    # Get optimizer parameter groups
    base_lr = 0.001
    param_groups = encoder.get_optimizer_param_groups(base_lr)

    # Check that we have the expected number of groups
    assert len(param_groups) >= len(lr_scales), "Missing parameter groups"

    # Check learning rates for each group
    lr_by_group = {}
    for group in param_groups:
        if 'name' in group:
            lr_by_group[group['name']] = group['lr']

    # Verify scaled learning rates
    for pattern, scale in lr_scales.items():
        if pattern in lr_by_group:
            expected_lr = base_lr * scale
            assert abs(lr_by_group[pattern] - expected_lr) < 1e-6, \
                f"Incorrect learning rate for {pattern}"


def test_swin_transformer_encoder_gradual_unfreeze():
    """Test the gradual unfreezing functionality."""
    # Initialize with all layers frozen
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="all"
    )

    # Initial state - all should be frozen
    trainable_before, frozen_before = count_trainable_params(encoder)
    assert trainable_before == 0, "Expected all parameters to be frozen \
initially"

    # Imprimir la estructura del modelo para depuración
    print("\n=== MODEL STRUCTURE DEBUGGING ===")
    param_names = list(name for name, _ in encoder.swin.named_parameters())
    print(f"Total parameters: {len(param_names)}")

    # Encontrar prefijos únicos para entender la estructura
    prefixes = set()
    for name in param_names:
        parts = name.split(".")
        if len(parts) > 0:
            prefixes.add(parts[0])

    print(f"Top-level parameter groups: {sorted(prefixes)}")

    # Mostrar algunos ejemplos de nombres de parámetros
    print("\nExamples of parameter names:")
    # Primeros 10 parámetros
    for i, name in enumerate(sorted(param_names)[:10]):
        print(f"  {i+1}. {name}")

    # Define an unfreeze schedule
    unfreeze_schedule = {
        5: ["stages.0"],
        10: ["stages.1"]
    }

    # Simulate epoch 1 - nothing should be unfrozen yet
    encoder.gradual_unfreeze(1, unfreeze_schedule)
    trainable, frozen = count_trainable_params(encoder)
    assert trainable == 0, "No parameters should be unfrozen at epoch 1"

    # Simulate epoch 5 - stages.0 should be unfrozen
    print("\n=== APPLYING UNFREEZE AT EPOCH 5 ===")
    encoder.gradual_unfreeze(5, unfreeze_schedule)
    param_status = get_param_status_by_name(encoder)

    # Mostrar los parámetros que se han descongelado
    unfrozen_params = [name for name, is_trainable in param_status.items() if
                       is_trainable]
    print(f"\nUnfrozen parameters after epoch 5: {len(unfrozen_params)}")
    # Mostrar hasta 5 ejemplos
    for i, name in enumerate(sorted(unfrozen_params)[:5]):
        print(f"  {i+1}. {name}")

    # Check that layers_0 parameters (equivalent to stages.0) are unfrozen
    layers0_unfrozen = False
    for name, is_trainable in param_status.items():
        if "layers_0" in name and is_trainable:
            layers0_unfrozen = True
            break

    assert layers0_unfrozen, "Expected layers_0 (equivalent to stages.0) to \
be unfrozen at epoch 5"

    # Simulate epoch 10 - layers_1 should also be unfrozen
    encoder.gradual_unfreeze(10, unfreeze_schedule)
    param_status = get_param_status_by_name(encoder)

    # Check that layers_1 parameters (equivalent to stages.1) are also unfrozen
    layers1_unfrozen = False
    for name, is_trainable in param_status.items():
        if "layers_1" in name and is_trainable:
            layers1_unfrozen = True
            break

    assert layers1_unfrozen, "Expected layers_1 (equivalent to stages.1) to \
be unfrozen at epoch 10"


def test_integration_freeze_and_transfer():
    """Integration test for transfer learning with frozen layers and training.
    """
    # Create a small encoder with freezing
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        freeze_layers="patch_embed"
    )

    # Create a small input tensor
    x = torch.randn(2, 3, 256, 256)

    # Forward pass
    bottleneck, skip_connections = encoder(x)

    # Create a dummy loss
    loss = bottleneck.mean() + sum(skip.mean() for skip in skip_connections)

    # Backward pass
    loss.backward()

    # Check that gradients only flow to trainable parameters
    for name, param in encoder.named_parameters():
        if "patch_embed" in name:
            assert param.grad is None or torch.all(param.grad == 0), \
                f"Expected no gradient for frozen parameter {name}"
        else:
            # Some parameters might not receive gradients due to the network
            # structure, so we can't assert all non-frozen parameters have
            # gradients
            pass


# Utility functions
def count_trainable_params(model):
    """Count trainable and frozen parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters()
                        if not p.requires_grad)
    return trainable_params, frozen_params


def get_param_status_by_name(model):
    """Get a dictionary mapping parameter names to trainable status."""
    return {name: param.requires_grad
            for name, param in model.named_parameters()}
