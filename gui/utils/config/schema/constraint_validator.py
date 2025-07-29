"""
Constraint validation for CrackSeg configurations.

This module provides validation for hardware constraints, domain-specific
requirements, and cross-parameter compatibility checks.
"""

from typing import Any

from ..exceptions import ValidationError


class ConstraintValidator:
    """Validator for hardware and domain-specific constraints."""

    def __init__(self) -> None:
        """Initialize constraint validator with CrackSeg-specific rules."""
        # Hardware constraints for RTX 3070 Ti
        self.hardware_constraints = {
            "max_vram_gb": 8,
            "recommended_batch_sizes": {
                "512x512": {"max": 16, "recommended": 8},
                "256x256": {"max": 32, "recommended": 16},
                "1024x1024": {"max": 4, "recommended": 2},
            },
        }

        # Domain-specific constraints for crack segmentation
        self.domain_constraints = {
            "min_image_size": 256,
            "max_image_size": 2048,
            "recommended_image_sizes": {256, 512, 1024},
            "max_rotation_angle": 45,
            "max_brightness_variation": 0.3,
        }

    def validate_hardware_constraints(
        self, training_config: dict[str, Any], data_config: dict[str, Any]
    ) -> list[str]:
        """Validate hardware constraints for RTX 3070 Ti."""
        warnings: list[str] = []

        batch_size = training_config.get("batch_size", 8)
        image_size = data_config.get("image_size", "512x512")

        if image_size in self.hardware_constraints["recommended_batch_sizes"]:
            max_batch = self.hardware_constraints["recommended_batch_sizes"][
                image_size
            ]["max"]
            recommended_batch = self.hardware_constraints[
                "recommended_batch_sizes"
            ][image_size]["recommended"]

            if batch_size > max_batch:
                warnings.append(
                    f"Batch size {batch_size} may exceed VRAM limits for "
                    f"{image_size} images on RTX 3070 Ti (max: {max_batch})"
                )
            elif batch_size > recommended_batch:
                warnings.append(
                    f"Batch size {batch_size} exceeds recommended value for "
                    f"{image_size} images (recommended: {recommended_batch})"
                )

        return warnings

    def validate_domain_constraints(
        self, data_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate domain-specific constraints for crack segmentation."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate image size constraints
        if "image_size" in data_config:
            image_size = data_config["image_size"]
            if isinstance(image_size, str):
                try:
                    width, height = map(int, image_size.split("x"))

                    # Check minimum size
                    if (
                        width < self.domain_constraints["min_image_size"]
                        or height < self.domain_constraints["min_image_size"]
                    ):
                        errors.append(
                            ValidationError(
                                f"Image size {image_size} is too small. "
                                f"Minimum size: "
                                f"{self.domain_constraints['min_image_size']}x"
                                f"{self.domain_constraints['min_image_size']}",
                                field="image_size",
                            )
                        )

                    # Check maximum size
                    if (
                        width > self.domain_constraints["max_image_size"]
                        or height > self.domain_constraints["max_image_size"]
                    ):
                        errors.append(
                            ValidationError(
                                f"Image size {image_size} is too large. "
                                f"Maximum size: "
                                f"{self.domain_constraints['max_image_size']}x"
                                f"{self.domain_constraints['max_image_size']}",
                                field="image_size",
                            )
                        )

                    # Check if size is recommended
                    if (
                        width
                        not in self.domain_constraints[
                            "recommended_image_sizes"
                        ]
                    ):
                        warnings.append(
                            f"Image width {width} is not a recommended size. "
                            f"Recommended sizes: "
                            f"{sorted(self.domain_constraints['recommended_image_sizes'])}"
                        )
                    if (
                        height
                        not in self.domain_constraints[
                            "recommended_image_sizes"
                        ]
                    ):
                        warnings.append(
                            f"Image height {height} is not a recommended size."
                            f" Recommended sizes: "
                            f"{sorted(self.domain_constraints['recommended_image_sizes'])}"
                        )

                except ValueError:
                    errors.append(
                        ValidationError(
                            "Invalid image size format. Use 'WIDTHxHEIGHT'",
                            field="image_size",
                        )
                    )

        # Validate augmentation constraints
        if "augmentation" in data_config:
            aug_config = data_config["augmentation"]
            if isinstance(aug_config, dict):
                # Validate rotation angle
                rotation = aug_config.get("rotation", 0)
                if rotation > self.domain_constraints["max_rotation_angle"]:
                    errors.append(
                        ValidationError(
                            f"Rotation angle {rotation} exceeds maximum "
                            "allowed value "
                            f"({self.domain_constraints['max_rotation_angle']})"
                            f" for crack detection",
                            field="augmentation.rotation",
                        )
                    )

                # Validate brightness variation
                brightness = aug_config.get("brightness", 0)
                if (
                    brightness
                    > self.domain_constraints["max_brightness_variation"]
                ):
                    errors.append(
                        ValidationError(
                            f"Brightness variation {brightness} exceeds "
                            "maximum allowed value "
                            f"({self.domain_constraints['max_brightness_variation']})"
                            " for crack visibility",
                            field="augmentation.brightness",
                        )
                    )

        return errors, warnings

    def validate_cross_parameter_constraints(
        self, config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate constraints between different configuration parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Check model output classes vs data configuration
        if "model" in config and "data" in config:
            model_config = config["model"]
            data_config = config["data"]

            model_classes = model_config.get("num_classes", 1)
            if "num_classes" in data_config:
                data_classes = data_config["num_classes"]
                if model_classes != data_classes:
                    errors.append(
                        ValidationError(
                            f"Model classes ({model_classes}) don't match "
                            f"data classes ({data_classes})",
                            field="num_classes",
                        )
                    )

        # Check batch size vs image size compatibility
        if "training" in config and "data" in config:
            training_config = config["training"]
            data_config = config["data"]

            batch_size = training_config.get("batch_size", 8)
            image_size = data_config.get("image_size", "512x512")

            if isinstance(image_size, str):
                try:
                    width, height = map(int, image_size.split("x"))
                    total_pixels = width * height

                    # Warn about large memory usage
                    if total_pixels * batch_size > 1000000:  # 1M pixels
                        warnings.append(
                            f"Large memory usage: {batch_size} batches of "
                            f"{image_size} images "
                            f"({total_pixels * batch_size:,} total pixels)"
                        )

                except ValueError:
                    pass

        return errors, warnings

    def validate_architecture_constraints(
        self, model_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate architecture-specific constraints."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        architecture = model_config.get("architecture", "")
        encoder = model_config.get("encoder", "")
        decoder = model_config.get("decoder", "")

        # Swin Transformer specific validations
        if "swin" in architecture.lower() and encoder != "swin_transformer":
            errors.append(
                ValidationError(
                    f"Swin architecture requires swin_transformer encoder, "
                    f"got {encoder}",
                    field="encoder",
                )
            )
        elif architecture == "cnn_lstm" and encoder != "cnn":
            errors.append(
                ValidationError(
                    f"CNN+LSTM architecture requires CNN encoder, got "
                    f"{encoder}",
                    field="encoder",
                )
            )
        elif architecture == "deeplabv3plus" and decoder != "deeplabv3plus":
            errors.append(
                ValidationError(
                    f"DeepLabV3+ architecture requires deeplabv3plus decoder, "
                    f"got {decoder}",
                    field="decoder",
                )
            )

        # U-Net specific validations
        if "unet" in architecture.lower() and decoder not in {
            "unet",
            "unetplusplus",
        }:
            warnings.append(
                "U-Net architecture typically uses U-Net decoder, got "
                f"{decoder}"
            )

        return errors, warnings

    def validate_training_constraints(
        self, training_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate training-specific constraints."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate learning rate
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if isinstance(lr, int | float):
                if lr > 1.0:
                    warnings.append(
                        f"High learning rate: {lr}. Consider reducing"
                    )
                elif lr < 1e-6:
                    warnings.append(
                        f"Very low learning rate: {lr}. Training may be slow"
                    )

        # Validate batch size
        if "batch_size" in training_config:
            batch_size = training_config["batch_size"]
            if isinstance(batch_size, int):
                if batch_size > 32:
                    warnings.append(
                        f"Large batch size: {batch_size}. "
                        "May cause VRAM issues on RTX 3070 Ti"
                    )
                elif batch_size < 2:
                    warnings.append(
                        f"Small batch size: {batch_size}. "
                        "May affect batch normalization and training stability"
                    )

        # Validate epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if isinstance(epochs, int):
                if epochs > 1000:
                    warnings.append(
                        f"High number of epochs: {epochs}. Consider early "
                        "stopping"
                    )
                elif epochs < 10:
                    warnings.append(
                        f"Low number of epochs: {epochs}. May not be "
                        "sufficient for convergence"
                    )

        return errors, warnings
