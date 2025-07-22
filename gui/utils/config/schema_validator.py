"""
Advanced Schema Validation System for Crack Segmentation Configurations.

This module provides comprehensive schema validation specifically tailored for
the CrackSeg pavement crack segmentation project. It includes deep validation
of model architectures, training parameters, data configurations, and ensures
compatibility with the project's specific requirements.

Key Features:
- Deep schema validation for crack segmentation models
- Architecture-specific validation (U-Net, CNN+LSTM, Swin Transformer)
- Hardware constraint validation (RTX 3070 Ti VRAM limits)
- Domain-specific validation (crack detection parameters)
- Type-safe configuration validation with detailed error reporting
"""

import logging
from collections.abc import Callable
from typing import Any

from .exceptions import ValidationError

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

        # Define validation rules
        self.validation_rules = self._initialize_validation_rules()

    def validate_complete_schema(
        self, config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Perform comprehensive schema validation on the entire configuration.

        Args:
            config: Complete configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate top-level structure
        structure_errors = self._validate_top_level_structure(config)
        errors.extend(structure_errors)

        # Validate individual sections
        for section_name in ["model", "training", "data"]:
            if section_name in config:
                section_errors, section_warnings = self._validate_section(
                    section_name, config[section_name], config
                )
                errors.extend(section_errors)
                warnings.extend(section_warnings)

        # Validate cross-section compatibility
        compatibility_errors, compatibility_warnings = (
            self._validate_cross_section_compatibility(config)
        )
        errors.extend(compatibility_errors)
        warnings.extend(compatibility_warnings)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_model_section(
        self, model_config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate model configuration section.

        Args:
            model_config: Model section of configuration

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate required fields
        required_fields = ["architecture", "encoder", "decoder"]
        for field in required_fields:
            if field not in model_config:
                errors.append(
                    ValidationError(
                        message=f"Missing required model field: '{field}'",
                        field=f"model.{field}",
                        suggestions=[
                            f"Add '{field}:' to your model configuration",
                            "See example model configs in configs/model/",
                        ],
                    )
                )

        # Validate architecture
        if "architecture" in model_config:
            arch_errors = self._validate_architecture(
                model_config["architecture"]
            )
            errors.extend(arch_errors)

        # Validate encoder
        if "encoder" in model_config:
            encoder_errors = self._validate_encoder(model_config["encoder"])
            errors.extend(encoder_errors)

        # Validate decoder
        if "decoder" in model_config:
            decoder_errors = self._validate_decoder(model_config["decoder"])
            errors.extend(decoder_errors)

        # Validate architecture compatibility
        if all(
            k in model_config for k in ["architecture", "encoder", "decoder"]
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

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_training_section(
        self,
        training_config: dict[str, Any],
        full_config: dict[str, Any] | None = None,
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate training configuration section.

        Args:
            training_config: Training section of configuration
            full_config: Complete configuration for cross-validation

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate required fields
        required_fields = ["epochs", "learning_rate", "optimizer"]
        for field in required_fields:
            if field not in training_config:
                errors.append(
                    ValidationError(
                        message=f"Missing required training field: '{field}'",
                        field=f"training.{field}",
                        suggestions=[
                            f"Add '{field}:' to your training configuration"
                        ],
                    )
                )

        # Validate training parameters
        param_errors, param_warnings = self._validate_training_parameters(
            training_config
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        # Validate optimizer configuration
        if "optimizer" in training_config:
            opt_errors = self._validate_optimizer_config(training_config)
            errors.extend(opt_errors)

        # Validate scheduler configuration
        if "scheduler" in training_config:
            sched_errors = self._validate_scheduler_config(training_config)
            errors.extend(sched_errors)

        # Validate loss function
        if "loss" in training_config:
            loss_errors = self._validate_loss_config(training_config["loss"])
            errors.extend(loss_errors)

        # Hardware-specific validation
        if full_config and "data" in full_config:
            hw_warnings = self._validate_hardware_constraints(
                training_config, full_config["data"]
            )
            warnings.extend(hw_warnings)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_data_section(
        self, data_config: dict[str, Any]
    ) -> tuple[bool, list[ValidationError], list[str]]:
        """
        Validate data configuration section.

        Args:
            data_config: Data section of configuration

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate required fields
        required_fields = ["data_root", "image_size", "batch_size"]
        for field in required_fields:
            if field not in data_config:
                errors.append(
                    ValidationError(
                        message=f"Missing required data field: '{field}'",
                        field=f"data.{field}",
                        suggestions=[
                            f"Add '{field}:' to your data configuration"
                        ],
                    )
                )

        # Validate data parameters
        param_errors, param_warnings = self._validate_data_parameters(
            data_config
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        # Validate crack-specific parameters
        crack_warnings = self._validate_crack_specific_parameters(data_config)
        warnings.extend(crack_warnings)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_top_level_structure(
        self, config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate top-level configuration structure."""
        errors: list[ValidationError] = []

        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                errors.append(
                    ValidationError(
                        message=(
                            "Missing required configuration section: "
                            f"'{section}'"
                        ),
                        field=section,
                        suggestions=[
                            f"Add the '{section}' key to your configuration "
                            "file."
                        ],
                    )
                )
            elif not isinstance(config[section], dict):
                msg = f"Configuration section '{section}' must be a dictionary"
                errors.append(ValidationError(message=msg, field=section))

        return errors

    def _validate_section(
        self,
        section_name: str,
        section_config: Any,
        full_config: dict[str, Any],
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate a specific configuration section."""
        if section_name == "model":
            _, errors, warnings = self.validate_model_section(section_config)
        elif section_name == "training":
            _, errors, warnings = self.validate_training_section(
                section_config, full_config
            )
        elif section_name == "data":
            _, errors, warnings = self.validate_data_section(section_config)
        else:
            errors, warnings = [], []

        return errors, warnings

    def _validate_architecture(
        self, architecture: Any
    ) -> list[ValidationError]:
        """Validate model architecture specification."""
        errors: list[ValidationError] = []

        if not isinstance(architecture, str):
            errors.append(
                ValidationError(
                    message="Model architecture must be a string",
                    field="model.architecture",
                )
            )
        elif architecture not in self.valid_architectures:
            errors.append(
                ValidationError(
                    message=f"Unknown architecture: '{architecture}'",
                    field="model.architecture",
                    suggestions=[
                        "Use one of: "
                        f"{', '.join(sorted(self.valid_architectures))}",
                        "Check configs/model/architectures/ for examples",
                    ],
                )
            )

        return errors

    def _validate_encoder(self, encoder: Any) -> list[ValidationError]:
        """Validate encoder specification."""
        errors: list[ValidationError] = []

        if not isinstance(encoder, str):
            errors.append(
                ValidationError(
                    message="Model encoder must be a string",
                    field="model.encoder",
                )
            )
        elif encoder not in self.valid_encoders:
            errors.append(
                ValidationError(
                    message=f"Unknown encoder: '{encoder}'",
                    field="model.encoder",
                    suggestions=[
                        "Use one of: "
                        f"{', '.join(sorted(self.valid_encoders))}",
                        "Check configs/model/encoder/ for examples",
                    ],
                )
            )

        return errors

    def _validate_decoder(self, decoder: Any) -> list[ValidationError]:
        """Validate decoder specification."""
        errors: list[ValidationError] = []

        if not isinstance(decoder, str):
            errors.append(
                ValidationError(
                    message="Model decoder must be a string",
                    field="model.decoder",
                )
            )
        elif decoder not in self.valid_decoders:
            errors.append(
                ValidationError(
                    message=f"Unknown decoder: '{decoder}'",
                    field="model.decoder",
                    suggestions=[
                        "Use one of: "
                        f"{', '.join(sorted(self.valid_decoders))}",
                        "Check configs/model/decoder/ for examples",
                    ],
                )
            )

        return errors

    def _validate_architecture_compatibility(
        self, architecture: str, encoder: str, decoder: str
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate compatibility between architecture components."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Check for known incompatible combinations
        incompatible_combinations = [
            ("swin_unet", "resnet50"),  # Swin U-Net with ResNet encoder
            ("cnn_convlstm_unet", "swin_transformer"),  # CNN+LSTM with Swin
        ]

        for incompatible_arch, incompatible_enc in incompatible_combinations:
            if (
                architecture == incompatible_arch
                and encoder == incompatible_enc
            ):
                errors.append(
                    ValidationError(
                        message=(
                            f"Incompatible combination: {architecture} with "
                            f"{encoder}"
                        ),
                        field="model",
                        suggestions=[
                            f"Use {incompatible_arch} with CNN or EfficientNet"
                            " encoders",
                            f"Use {incompatible_enc} with U-Net or DeepLabV3+ "
                            "architectures",
                        ],
                    )
                )

        # Add performance warnings for heavy combinations
        heavy_combinations = [
            ("swin_unet", "swin_transformer"),
            ("unet_aspp", "resnet152"),
        ]

        for heavy_arch, heavy_enc in heavy_combinations:
            if architecture == heavy_arch and encoder == heavy_enc:
                warnings.append(
                    f"Performance warning: {architecture} + {encoder} "
                    "may exceed RTX 3070 Ti VRAM limits with large batch sizes"
                )

        return errors, warnings

    def _validate_model_parameters(
        self, model_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate model-specific parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate num_classes
        if "num_classes" in model_config:
            num_classes = model_config["num_classes"]
            if not isinstance(num_classes, int) or num_classes < 1:
                errors.append(
                    ValidationError(
                        message="num_classes must be a positive integer",
                        field="model.num_classes",
                    )
                )
            elif num_classes != 1:
                warnings.append(
                    "CrackSeg is designed for binary segmentation "
                    "(num_classes=1). Verify your configuration is correct."
                )

        # Validate input_channels
        if "input_channels" in model_config:
            input_channels = model_config["input_channels"]
            if not isinstance(input_channels, int) or input_channels < 1:
                errors.append(
                    ValidationError(
                        message="input_channels must be a positive integer",
                        field="model.input_channels",
                    )
                )
            elif input_channels != 3:
                warnings.append(
                    "Most crack segmentation models expect RGB input "
                    "(input_channels=3). Verify your data format."
                )
        return errors, warnings

    def _validate_training_parameters(
        self, training_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate training-specific parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate epochs
        if "epochs" in training_config:
            epochs = training_config["epochs"]
            if not isinstance(epochs, int) or epochs < 1:
                errors.append(
                    ValidationError(
                        message="epochs must be a positive integer",
                        field="training.epochs",
                    )
                )
            elif epochs < 10:
                warnings.append(
                    f"Training for only {epochs} epochs may be insufficient "
                    "for crack segmentation convergence"
                )

        # Validate learning_rate
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if not isinstance(lr, int | float) or lr <= 0:
                errors.append(
                    ValidationError(
                        message="learning_rate must be a positive number",
                        field="training.learning_rate",
                    )
                )
            elif lr > 0.1:
                warnings.append(
                    f"Learning rate {lr} is very high for crack segmentation. "
                    f"Consider values between 1e-5 and 1e-2"
                )

        return errors, warnings

    def _validate_optimizer_config(
        self, training_config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate optimizer configuration."""
        errors: list[ValidationError] = []

        optimizer = training_config.get("optimizer")
        if isinstance(optimizer, str):
            if optimizer not in self.valid_optimizers:
                errors.append(
                    ValidationError(
                        message=f"Unknown optimizer: '{optimizer}'",
                        field="training.optimizer",
                        suggestions=[
                            "Use one of: "
                            f"{', '.join(sorted(self.valid_optimizers))}"
                        ],
                    )
                )
        elif isinstance(optimizer, dict):
            if "_target_" not in optimizer:
                errors.append(
                    ValidationError(
                        message="Optimizer config must have '_target_' field",
                        field="training.optimizer",
                    )
                )

        return errors

    def _validate_scheduler_config(
        self, training_config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate learning rate scheduler configuration."""
        errors: list[ValidationError] = []

        scheduler = training_config.get("scheduler")
        if isinstance(scheduler, str):
            if scheduler not in self.valid_schedulers:
                errors.append(
                    ValidationError(
                        message=f"Unknown scheduler: '{scheduler}'",
                        field="training.scheduler",
                        suggestions=[
                            "Use one of: "
                            f"{', '.join(sorted(self.valid_schedulers))}"
                        ],
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
                        message=f"Unknown loss function: '{loss_config}'",
                        field="training.loss",
                        suggestions=[
                            "Use one of: "
                            f"{', '.join(sorted(self.valid_loss_functions))}"
                        ],
                    )
                )
        elif isinstance(loss_config, dict):
            if "_target_" not in loss_config:
                errors.append(
                    ValidationError(
                        message="Loss config must have '_target_' field",
                        field="training.loss",
                    )
                )

        return errors

    def _validate_hardware_constraints(
        self, training_config: dict[str, Any], data_config: dict[str, Any]
    ) -> list[str]:
        """Validate configuration against hardware constraints."""
        warnings: list[str] = []

        batch_size = training_config.get("batch_size") or data_config.get(
            "batch_size"
        )
        image_size = data_config.get("image_size")

        if batch_size and image_size:
            if isinstance(image_size, list) and len(image_size) == 2:
                size_key = f"{image_size[0]}x{image_size[1]}"
                batch_size_constraints = self.hardware_constraints[
                    "recommended_batch_sizes"
                ]
                if (
                    isinstance(batch_size_constraints, dict)
                    and size_key in batch_size_constraints
                ):
                    limits = batch_size_constraints[size_key]
                    if batch_size > limits["max"]:
                        warnings.append(
                            f"Batch size {batch_size} may exceed RTX 3070 Ti "
                            f"VRAM for {size_key} images. Max recommended: "
                            f"{limits['max']}"
                        )
                    elif batch_size > limits["recommended"]:
                        warnings.append(
                            f"Batch size {batch_size} is above recommended "
                            f"for {size_key} images. Recommended: "
                            f"{limits['recommended']}"
                        )

        return warnings

    def _validate_data_parameters(
        self, data_config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate data-specific parameters."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate image_size
        if "image_size" in data_config:
            image_size = data_config["image_size"]
            if not isinstance(image_size, list) or len(image_size) != 2:
                errors.append(
                    ValidationError(
                        message=(
                            "image_size must be a list of two integers "
                            "[height, width]"
                        ),
                        field="data.image_size",
                    )
                )
            elif not all(isinstance(x, int) and x > 0 for x in image_size):
                errors.append(
                    ValidationError(
                        message="image_size values must be positive integers",
                        field="data.image_size",
                    )
                )

        # Validate batch_size
        if "batch_size" in data_config:
            batch_size = data_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append(
                    ValidationError(
                        message="batch_size must be a positive integer",
                        field="data.batch_size",
                    )
                )

        return errors, warnings

    def _validate_crack_specific_parameters(
        self, data_config: dict[str, Any]
    ) -> list[str]:
        """Validate crack segmentation specific parameters."""
        warnings: list[str] = []

        # Check for appropriate augmentations
        if "augmentations" in data_config:
            augs = data_config["augmentations"]
            if isinstance(augs, dict) and "rotation" in augs:
                rotation = augs.get("rotation", {})
                if isinstance(rotation, dict):
                    limit = rotation.get("limit", 0)
                    if limit > 45:
                        warnings.append(
                            f"Rotation augmentation limit {limit}° may be too "
                            "aggressive for crack detection. Consider limiting"
                            " to 15-30°"
                        )

        return warnings

    def _validate_cross_section_compatibility(
        self, config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate compatibility across configuration sections."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Check batch size consistency
        training_batch = config.get("training", {}).get("batch_size")
        data_batch = config.get("data", {}).get("batch_size")

        if training_batch and data_batch and training_batch != data_batch:
            warnings.append(
                f"Batch size mismatch: training={training_batch}, "
                f"data={data_batch}. Data batch size will be used."
            )

        return errors, warnings

    def _initialize_validation_rules(self) -> dict[str, Callable[[Any], bool]]:
        """Initialize validation rules for common parameter types."""
        return {
            "positive_int": lambda x: isinstance(x, int) and x > 0,
            "positive_float": lambda x: isinstance(x, int | float) and x > 0,
            "probability": lambda x: isinstance(x, int | float)
            and 0 <= x <= 1,
            "string": lambda x: isinstance(x, str),
            "list": lambda x: isinstance(x, list),
            "dict": lambda x: isinstance(x, dict),
        }


# Global validator instance
_schema_validator = CrackSegSchemaValidator()


def validate_crackseg_schema(
    config: dict[str, Any],
) -> tuple[bool, list[ValidationError], list[str]]:
    """
    Convenience function for validating crack segmentation schemas.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    return _schema_validator.validate_complete_schema(config)
