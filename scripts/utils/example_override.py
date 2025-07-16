"""
Example script: How to use the configuration system with overrides

This script demonstrates how to create a base configuration, apply command-line
style overrides, and print the resulting configuration in YAML format. This is
useful for testing and for users who want to understand how to modify
configuration values dynamically.
"""

import os

from omegaconf import OmegaConf

# from crackseg.utils.config_override import apply_overrides
# Module not found


def write_examples_to_file() -> None:
    """Write Hydra override examples to a file."""
    examples = [
        (1, "Basic value override", "python -m src.main training.epochs=100"),
        (
            2,
            "Multiple overrides",
            "python -m src.main training.epochs=100 "
            "model.encoder.name=resnet50",
        ),
        (
            3,
            "Override nested values",
            "python -m src.main model.encoder.pretrained=false "
            "model.decoder.channels=[256,128,64]",
        ),
        (
            4,
            "Override with different config group",
            "python -m src.main model=unet_resnet training=fast",
        ),
        (
            5,
            "Override with list values",
            "python -m src.main data.transforms=[resize,normalize,augment]",
        ),
        (
            6,
            "Override with null value",
            "python -m src.main training.scheduler=null",
        ),
        (
            7,
            "Override with complex nested structure",
            "python -m src.main 'model.encoder={name:resnet34,pretrained:true,"
            "channels:[64,128,256]}'",
        ),
        (
            8,
            "Override output directory",
            "python -m src.main hydra.run.dir=outputs/custom_run",
        ),
        (
            9,
            "Override with environment variables",
            "python -m src.main +training.device=${CUDA_VISIBLE_DEVICES}",
        ),
    ]

    # Create scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)

    # Write examples to file
    with open("scripts/hydra_examples.txt", "w") as f:
        f.write("Hydra Override Examples:\n")
        for num, title, command in examples:
            f.write(f"\n{num}. {title}:\n")
            f.write(f"{command}\n")


def main() -> None:
    """Run example configuration overrides."""
    # Base configuration (could be loaded from file or defined inline)
    base_config = {
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "model": {
            "name": "unet",
            "encoder": {"name": "resnet18", "pretrained": True},
        },
    }

    # Create a DictConfig object from the base dictionary
    cfg = OmegaConf.create(base_config)

    print("\nOriginal configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Example overrides (as would be passed from the command line)

    # Apply the overrides to the configuration
    # cfg_overridden = apply_overrides(cfg, overrides)  # Function not avail.
    cfg_overridden = cfg  # Placeholder

    print("\nConfiguration after applying overrides:")
    print(OmegaConf.to_yaml(cfg_overridden))

    # Write examples to file
    write_examples_to_file()

    # Display examples from file
    print("\nExamples have been written to scripts/hydra_examples.txt")
    with open("scripts/hydra_examples.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
