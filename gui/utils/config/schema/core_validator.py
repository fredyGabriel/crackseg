"""
Core schema validator for CrackSeg configurations.

This module contains the main validation logic for crack segmentation
configurations, including model architecture validation and cross-section
compatibility checks.
"""

import logging
from typing import Any

from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class CrackSegSchemaValidator:
    """
    Advanced schema validator for crack segmentation configurations.

    Provides comprehensive validation of configuration schemas with
    specific knowledge of crack segmentation model requirements,
    hardware constraints, and domain-specific parameters.
    """

    def __init__(self) -> None:
        """Initialize the schema validator with CrackSeg-specific rules."""
        # Define valid architecture combinations
        self.valid_architectures = {
            "unet",
            "cnn_convlstm_unet",
            "swin_unet",
            "unet_aspp",
            "deeplabv3plus",
            "fcn",
            "segnet",
        }

        self.valid_encoders = {
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "swin_transformer",
            "cnn",
            "mobilenet_v3",
            "vgg16",
        }

        self.valid_decoders = {
            "unet",
            "fpn",
            "unetplusplus",
            "deeplabv3plus",
            "pan",
        }

        self.valid_loss_functions = {
            "dice",
            "focal",
            "bce",
            "combined",
            "weighted_dice",
            "tversky",
            "focal_tversky",
            "boundary_loss",
        }

        self.valid_optimizers = {"adam", "adamw", "sgd", "rmsprop", "adagrad"}

        self.valid_schedulers = {
            "cosine",
            "step",
            "exponential",
            "plateau",
            "cyclic",
        }

        # Hardware constraints for RTX 3070 Ti
        self.hardware_constraints = {
            "max_vram_gb": 8,
            "recommended_batch_sizes": {
                "512x512": {"max": 16, "recommended": 8},
                "256x256": {"max": 32, "recommended": 16},
                "1024x1024": {"max": 4, "recommended": 2},
            },
        }

    def validate_complete_schema(
        self, config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate complete configuration schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate top-level structure
        errors.extend(self._validate_top_level_structure(config))

        # Validate model section
        if "model" in config:
            _, model_errors, model_warnings = self.validate_model_section(
                config["model"]
            )
            errors.extend(model_errors)
            warnings.extend(model_warnings)

        # Validate training section
        if "training" in config:
            _, training_errors, training_warnings = (
                self.validate_training_section(config["training"], config)
            )
            errors.extend(training_errors)
            warnings.extend(training_warnings)

        # Validate data section
        if "data" in config:
            _, data_errors, data_warnings = self.validate_data_section(
                config["data"]
            )
            errors.extend(data_errors)
            warnings.extend(data_warnings)

        # Validate cross-section compatibility
        cross_errors, cross_warnings = (
            self._validate_cross_section_compatibility(config)
        )
        errors.extend(cross_errors)
        warnings.extend(cross_warnings)

        return len(errors) == 0, errors, warnings

    def validate_model_section(
        self, model_config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate model configuration section.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate architecture
        if "architecture" in model_config:
            architecture = model_config["architecture"]
            if not isinstance(architecture, str):
                errors.append(
                    ValidationError(
                        "Architecture must be a string",
                        field="architecture",
                    )
                )
            elif architecture not in self.valid_architectures:
                errors.append(
                    ValidationError(
                        f"Invalid architecture: {architecture}. "
                        f"Valid options: {sorted(self.valid_architectures)}",
                        field="architecture",
                    )
                )

        # Validate encoder
        if "encoder" in model_config:
            encoder = model_config["encoder"]
            if not isinstance(encoder, str):
                errors.append(
                    ValidationError(
                        "Encoder must be a string",
                        field="encoder",
                    )
                )
            elif encoder not in self.valid_encoders:
                errors.append(
                    ValidationError(
                        f"Invalid encoder: {encoder}. "
                        f"Valid options: {sorted(self.valid_encoders)}",
                        field="encoder",
                    )
                )

        # Validate decoder
        if "decoder" in model_config:
            decoder = model_config["decoder"]
            if not isinstance(decoder, str):
                errors.append(
                    ValidationError(
                        "Decoder must be a string",
                        field="decoder",
                    )
                )
            elif decoder not in self.valid_decoders:
                errors.append(
                    ValidationError(
                        f"Invalid decoder: {decoder}. "
                        f"Valid options: {sorted(self.valid_decoders)}",
                        field="decoder",
                    )
                )

        # Validate architecture compatibility
        if all(
            key in model_config
            for key in ["architecture", "encoder", "decoder"]
        ):
            compat_errors, compat_warnings = (
                self._validate_architecture_compatibility(
                    model_config["architecture"],
                    model_config["encoder"],
                    model_config["decoder"],
                )
            )
            errors.extend(compat_errors)
            warnings.extend(compat_warnings)

        # Validate model parameters
        param_errors, param_warnings = self._validate_model_parameters(
            model_config
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        return len(errors) == 0, errors, warnings

    def validate_training_section(
        self,
        training_config: dict[str, Any],
        full_config: dict[str, Any] | None = None,
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate training configuration section.

        Args:
            training_config: Training configuration dictionary
            full_config: Complete configuration for cross-validation

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate training parameters
        param_errors, param_warnings = self._validate_training_parameters(
            training_config
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        # Validate optimizer configuration
        errors.extend(self._validate_optimizer_config(training_config))

        # Validate scheduler configuration
        errors.extend(self._validate_scheduler_config(training_config))

        # Validate loss configuration
        if "loss" in training_config:
            errors.extend(self._validate_loss_config(training_config["loss"]))

        # Validate hardware constraints
        if full_config and "data" in full_config:
            hw_warnings = self._validate_hardware_constraints(
                training_config, full_config["data"]
            )
            warnings.extend(hw_warnings)

        return len(errors) == 0, errors, warnings

    def validate_data_section(
        self, data_config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate data configuration section.

        Args:
            data_config: Data configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate data parameters
        param_errors, param_warnings = self._validate_data_parameters(
            data_config
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        # Validate crack-specific parameters
        crack_warnings = self._validate_crack_specific_parameters(data_config)
        warnings.extend(crack_warnings)

        return len(errors) == 0, errors, warnings

    def _validate_top_level_structure(
        self, config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate top-level configuration structure."""
        errors: list[ValidationError] = []

        required_sections = {"model", "training", "data"}
        missing_sections = required_sections - set(config.keys())

        for section in missing_sections:
            errors.append(
                ValidationError(
                    f"Missing required section: {section}",
                    field=section,
                )
            )

        return errors

    def _validate_architecture(
        self, architecture: Any
    ) -> list[ValidationError]:
        """Validate model architecture."""
        errors: list[ValidationError] = []

        if not isinstance(architecture, str):
            errors.append(
                ValidationError(
                    "Architecture must be a string",
                    field="architecture",
                )
            )
        elif architecture not in self.valid_architectures:
            errors.append(
                ValidationError(
                    f"Invalid architecture: {architecture}. "
                    f"Valid options: {sorted(self.valid_architectures)}",
                    field="architecture",
                )
            )

        return errors

    def _validate_encoder(self, encoder: Any) -> list[ValidationError]:
        """Validate encoder configuration."""
        errors: list[ValidationError] = []

        if not isinstance(encoder, str):
            errors.append(
                ValidationError(
                    "Encoder must be a string",
                    field="encoder",
                )
            )
        elif encoder not in self.valid_encoders:
            errors.append(
                ValidationError(
                    f"Invalid encoder: {encoder}. "
                    f"Valid options: {sorted(self.valid_encoders)}",
                    field="encoder",
                )
            )

        return errors

    def _validate_decoder(self, decoder: Any) -> list[ValidationError]:
        """Validate decoder configuration."""
        errors: list[ValidationError] = []

        if not isinstance(decoder, str):
            errors.append(
                ValidationError(
                    "Decoder must be a string",
                    field="decoder",
                )
            )
        elif decoder not in self.valid_decoders:
            errors.append(
                ValidationError(
                    f"Invalid decoder: {decoder}. "
                    f"Valid options: {sorted(self.valid_decoders)}",
                    field="decoder",
                )
            )

        return errors

    def _validate_architecture_compatibility(
        self, architecture: str, encoder: str, decoder: str
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate architecture, encoder, and decoder compatibility."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

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

        return errors, warnings

    def _validate_model_parameters(
        self, model_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate model-specific parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate input channels
        if "input_channels" in model_config:
            channels = model_config["input_channels"]
            if not isinstance(channels, int) or channels <= 0:
                errors.append(
                    ValidationError(
                        "Input channels must be a positive integer",
                        field="input_channels",
                    )
                )
            elif channels != 3:
                warnings.append(
                    f"Non-standard input channels: {channels}. "
                    "Crack segmentation typically uses 3 channels (RGB)"
                )

        # Validate output classes
        if "num_classes" in model_config:
            classes = model_config["num_classes"]
            if not isinstance(classes, int) or classes <= 0:
                errors.append(
                    ValidationError(
                        "Number of classes must be a positive integer",
                        field="num_classes",
                    )
                )
            elif classes != 1:
                warnings.append(
                    f"Non-standard number of classes: {classes}. "
                    "Binary crack segmentation typically uses 1 class"
                )

        return errors, warnings

    def _validate_training_parameters(
        self, training_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate training parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate learning rate
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if not isinstance(lr, int | float) or lr <= 0:
                errors.append(
                    ValidationError(
                        "Learning rate must be a positive number",
                        field="learning_rate",
                    )
                )
            elif lr > 1.0:
                warnings.append(f"High learning rate: {lr}. Consider reducing")

        # Validate batch size
        if "batch_size" in training_config:
            batch_size = training_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(
                    ValidationError(
                        "Batch size must be a positive integer",
                        field="batch_size",
                    )
                )
            elif batch_size > 32:
                warnings.append(
                    f"Large batch size: {batch_size}. "
                    "May cause VRAM issues on RTX 3070 Ti"
                )

        # Validate epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append(
                    ValidationError(
                        "Epochs must be a positive integer",
                        field="epochs",
                    )
                )
            elif epochs > 1000:
                warnings.append(
                    f"High number of epochs: {epochs}. Consider early stopping"
                )

        return errors, warnings

    def _validate_optimizer_config(
        self, training_config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate optimizer configuration."""
        errors: list[ValidationError] = []

        if "optimizer" in training_config:
            optimizer = training_config["optimizer"]
            if not isinstance(optimizer, str):
                errors.append(
                    ValidationError(
                        "Optimizer must be a string",
                        field="optimizer",
                    )
                )
            elif optimizer not in self.valid_optimizers:
                errors.append(
                    ValidationError(
                        f"Invalid optimizer: {optimizer}. "
                        f"Valid options: {sorted(self.valid_optimizers)}",
                        field="optimizer",
                    )
                )

        return errors

    def _validate_scheduler_config(
        self, training_config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate scheduler configuration."""
        errors: list[ValidationError] = []

        if "scheduler" in training_config:
            scheduler = training_config["scheduler"]
            if not isinstance(scheduler, str):
                errors.append(
                    ValidationError(
                        "Scheduler must be a string",
                        field="scheduler",
                    )
                )
            elif scheduler not in self.valid_schedulers:
                errors.append(
                    ValidationError(
                        f"Invalid scheduler: {scheduler}. "
                        f"Valid options: {sorted(self.valid_schedulers)}",
                        field="scheduler",
                    )
                )

        return errors

    def _validate_loss_config(self, loss_config: Any) -> list[ValidationError]:
        """Validate loss function configuration."""
        errors: list[ValidationError] = []

        if isinstance(loss_config, str):
            if loss_config not in self.valid_loss_functions:
                errors.append(
                    ValidationError(
                        f"Invalid loss function: {loss_config}. "
                        f"Valid options: {sorted(self.valid_loss_functions)}",
                        field="loss",
                    )
                )
        elif isinstance(loss_config, dict):
            if "name" in loss_config:
                if loss_config["name"] not in self.valid_loss_functions:
                    errors.append(
                        ValidationError(
                            f"Invalid loss function: {loss_config['name']}. "
                            f"Valid options: "
                            f"{sorted(self.valid_loss_functions)}",
                            field="loss.name",
                        )
                    )
        else:
            errors.append(
                ValidationError(
                    "Loss must be a string or dictionary with 'name' field",
                    field="loss",
                )
            )

        return errors

    def _validate_hardware_constraints(
        self, training_config: dict[str, Any], data_config: dict[str, Any]
    ) -> list[str]:
        """Validate hardware constraints for RTX 3070 Ti."""
        warnings: list[str] = []

        batch_size = training_config.get("batch_size", 8)
        image_size = data_config.get("image_size", "512x512")

        # Normalize image_size to string key (e.g., "512x512") for lookup
        if isinstance(image_size, list | tuple):
            try:
                w, h = image_size
                image_size_key = f"{int(w)}x{int(h)}"
            except Exception:
                image_size_key = "512x512"
        elif isinstance(image_size, str):
            image_size_key = image_size
        else:
            image_size_key = "512x512"

        if (
            image_size_key
            in self.hardware_constraints["recommended_batch_sizes"]
        ):
            max_batch = self.hardware_constraints["recommended_batch_sizes"][
                image_size_key
            ]["max"]
            if batch_size > max_batch:
                warnings.append(
                    f"Batch size {batch_size} may exceed VRAM limits for "
                    f"{image_size} images on RTX 3070 Ti (max: {max_batch})"
                )

        return warnings

    def _validate_data_parameters(
        self, data_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate data parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate image size
        if "image_size" in data_config:
            image_size = data_config["image_size"]
            if not isinstance(image_size, str):
                errors.append(
                    ValidationError(
                        "Image size must be a string (e.g., '512x512')",
                        field="image_size",
                    )
                )
            else:
                try:
                    width, height = map(int, image_size.split("x"))
                    if width <= 0 or height <= 0:
                        errors.append(
                            ValidationError(
                                "Image dimensions must be positive",
                                field="image_size",
                            )
                        )
                    elif width != height:
                        warnings.append(
                            f"Non-square image size: {image_size}. "
                            "Square images are recommended for crack "
                            "segmentation"
                        )
                except ValueError:
                    errors.append(
                        ValidationError(
                            "Invalid image size format. Use 'WIDTHxHEIGHT'",
                            field="image_size",
                        )
                    )

        # Validate data paths
        required_paths = ["train_path", "val_path", "test_path"]
        for path_key in required_paths:
            if path_key in data_config:
                path = data_config[path_key]
                if not isinstance(path, str) or not path.strip():
                    errors.append(
                        ValidationError(
                            f"{path_key} must be a non-empty string",
                            field=path_key,
                        )
                    )

        return errors, warnings

    def _validate_crack_specific_parameters(
        self, data_config: dict[str, Any]
    ) -> list[str]:
        """Validate crack-specific parameters."""
        warnings: list[str] = []

        # Validate augmentation parameters
        if "augmentation" in data_config:
            aug_config = data_config["augmentation"]
            if isinstance(aug_config, dict):
                if aug_config.get("rotation", 0) > 45:
                    warnings.append(
                        "High rotation angle may affect crack orientation "
                        "detection"
                    )
                if aug_config.get("brightness", 0) > 0.3:
                    warnings.append(
                        "High brightness variation may affect crack visibility"
                    )

        return warnings

    def _validate_cross_section_compatibility(
        self, config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate compatibility between different configuration sections."""
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

        return errors, warnings
