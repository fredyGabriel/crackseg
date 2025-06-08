"""Examples and demonstrations of advanced override parsing capabilities.

This module provides examples of how to use the AdvancedOverrideParser
for complex Hydra configuration overrides in the CrackSeg project.
"""

from .parsing import AdvancedOverrideParser
from .run_manager import get_process_manager


def demonstrate_override_parsing() -> None:
    """Demonstrate various override parsing capabilities."""
    parser = AdvancedOverrideParser()

    print("=== Advanced Override Parser Demonstration ===\n")

    # Example 1: Basic configuration overrides
    print("1. Basic Configuration Overrides:")
    basic_overrides = """
    trainer.max_epochs=100
    model.encoder=resnet50
    training.learning_rate=0.001
    """

    parser.parse_overrides(basic_overrides)
    print(f"Input: {basic_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")

    # Example 2: Complex nested configurations
    print("2. Complex Nested Configurations:")
    complex_overrides = """
    model.decoder.attention.num_heads=8
    training.optimizer.weight_decay=1e-4
    data.augmentation.rotation_limit=15
    """

    parser.parse_overrides(complex_overrides)
    print(f"Input: {complex_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")

    # Example 3: Package and force overrides
    print("3. Package and Force Overrides:")
    package_overrides = """
    +model/encoder=swin_transformer
    ++training.device=cuda:0
    ~model.pretrained
    """

    parser.parse_overrides(package_overrides)
    print(f"Input: {package_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")

    # Example 4: List and complex values
    print("4. List and Complex Values:")
    list_overrides = """
    training.batch_sizes=[4,8,16]
    model.channels=[64,128,256,512]
    data.mean=[0.485,0.456,0.406]
    """

    parser.parse_overrides(list_overrides)
    print(f"Input: {list_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")

    # Example 5: Quoted strings and special characters
    print("5. Quoted Strings and Special Characters:")
    quoted_overrides = """
    experiment.name="crack_segmentation_v2"
    training.checkpoint_dir="/path/to/checkpoints"
    model.description='ResNet50 encoder with U-Net decoder'
    """

    parser.parse_overrides(quoted_overrides)
    print(f"Input: {quoted_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")

    # Example 6: Invalid overrides (security demonstration)
    print("6. Invalid/Dangerous Overrides (Security Check):")
    dangerous_overrides = """
    model.encoder=resnet50; rm -rf /
    training.command=`cat /etc/passwd`
    data.path=../../../etc/hosts
    """

    parser.parse_overrides(dangerous_overrides)
    print(f"Input: {dangerous_overrides.strip()}")
    print(f"Valid overrides: {parser.get_valid_overrides()}")
    print(f"Errors: {parser.get_parsing_errors()}\n")


def demonstrate_process_manager_integration() -> None:
    """Demonstrate integration with ProcessManager."""
    print("=== ProcessManager Integration Demonstration ===\n")

    manager = get_process_manager()

    # Example 1: Parse override text
    print("1. Parsing Override Text:")
    override_text = """
    trainer.max_epochs=50
    model.encoder=resnet34
    training.batch_size=8
    +experiment=crack_detection
    """

    valid_overrides, errors = manager.parse_overrides_text(override_text)
    print(f"Input text: {override_text.strip()}")
    print(f"Valid overrides: {valid_overrides}")
    print(f"Parsing errors: {errors}\n")

    # Example 2: Validate single overrides
    print("2. Single Override Validation:")
    test_overrides = [
        "trainer.max_epochs=100",
        "invalid..key=value",
        "model.encoder=resnet50",
        "dangerous; rm -rf /",
        "+model/decoder=unet",
    ]

    for override in test_overrides:
        is_valid, error = manager.validate_single_override(override)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{status}: {override}")
        if error:
            print(f"   Error: {error}")
    print()


def get_crackseg_override_examples() -> dict[str, list[str]]:
    """Get common override examples for CrackSeg project.

    Returns:
        Dictionary of override categories with example lists
    """
    return {
        "training_configs": [
            "trainer.max_epochs=100",
            "trainer.early_stopping_patience=10",
            "training.learning_rate=1e-4",
            "training.batch_size=16",
            "training.gradient_clip_val=1.0",
        ],
        "model_architectures": [
            "model.encoder=resnet50",
            "model.encoder=swin_transformer_tiny",
            "model.decoder=unet",
            "model.decoder=deeplabv3plus",
            "+model/encoder=efficientnet_b0",
        ],
        "data_configurations": [
            "data.image_size=512",
            "data.num_workers=4",
            "data.pin_memory=true",
            "data.augmentation.rotation_limit=15",
            "data.augmentation.brightness_limit=0.2",
        ],
        "loss_functions": [
            "training.loss=dice_loss",
            "training.loss=focal_dice_loss",
            "training.loss_weights=[1.0,2.0]",
            "+training/loss=boundary_aware_loss",
            "training.loss.smooth=1e-5",
        ],
        "optimization": [
            "training.optimizer=adam",
            "training.optimizer.weight_decay=1e-4",
            "training.scheduler=cosine_annealing",
            "training.scheduler.T_max=100",
            "++training.use_amp=true",
        ],
        "experiment_tracking": [
            "experiment.name=crack_segmentation_v1",
            "experiment.tags=[baseline,resnet50]",
            "logging.log_every_n_steps=10",
            "logging.save_top_k=3",
            "hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        ],
        "hardware_specific": [
            "training.device=cuda:0",
            "training.precision=16",  # Mixed precision for RTX 3070 Ti
            "data.batch_size=4",  # Optimized for 8GB VRAM
            "training.accumulate_grad_batches=4",
            "training.enable_checkpointing=true",
        ],
    }


def validate_crackseg_examples() -> None:
    """Validate all CrackSeg override examples."""
    print("=== CrackSeg Override Examples Validation ===\n")

    parser = AdvancedOverrideParser()
    examples = get_crackseg_override_examples()

    for category, overrides in examples.items():
        print(f"{category.upper().replace('_', ' ')}:")

        for override in overrides:
            is_valid, error = parser.validate_override_string(override)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {override}")
            if error:
                print(f"    Error: {error}")
        print()


if __name__ == "__main__":
    """Run demonstrations when script is executed directly."""
    demonstrate_override_parsing()
    print("\n" + "=" * 60 + "\n")
    demonstrate_process_manager_integration()
    print("\n" + "=" * 60 + "\n")
    validate_crackseg_examples()
