"""Core YAML validation engine.

This module contains the main YAMLValidator class that provides comprehensive
validation of YAML configuration files with detailed error reporting and
intelligent suggestions for fixes.
"""

import yaml

from ..exceptions import ValidationError


class YAMLValidator:
    """Advanced YAML validation engine for configuration files."""

    def __init__(self) -> None:
        """Initialize the YAML validator."""
        # Common schema patterns for crack segmentation project
        self.model_architectures = {
            "unet",
            "cnn_convlstm_unet",
            "swin_unet",
            "unet_aspp",
        }
        self.encoders = {
            "resnet50",
            "resnet101",
            "efficientnet",
            "swin_transformer",
            "cnn",
        }
        self.loss_functions = {
            "dice",
            "focal",
            "bce",
            "combined",
            "weighted_dice",
        }
        self.optimizers = {"adam", "sgd", "adamw", "rmsprop"}
        self.schedulers = {"cosine", "step", "exponential", "plateau"}

    def validate_syntax(
        self, content: str
    ) -> tuple[bool, ValidationError | None]:
        """Validate YAML syntax with detailed error reporting.

        Args:
            content: YAML content as string.

        Returns:
            Tuple of (is_valid, validation_error).
        """
        try:
            yaml.safe_load(content)
            return True, None
        except yaml.YAMLError as e:
            # Extract detailed error information
            line = getattr(e, "problem_mark", None)
            problem = getattr(e, "problem", None)
            context = getattr(e, "context", None)

            line_num = getattr(line, "line", 0) + 1 if line else None
            col_num = getattr(line, "column", 0) + 1 if line else None

            message = str(problem) if problem else str(e)
            if context:
                message = f"{context}: {message}"

            # Provide suggestions based on common errors
            suggestions = self._get_syntax_suggestions(
                message, content, line_num
            )

            return False, ValidationError(
                message=f"YAML syntax error: {message}",
                line=line_num,
                column=col_num,
                suggestions=suggestions,
            )
        except Exception as e:
            return False, ValidationError(
                message=f"Unexpected error during YAML parsing: {str(e)}"
            )

    def validate_structure(
        self, config: dict[str, object]
    ) -> tuple[bool, list[ValidationError]]:
        """Validate configuration structure and required fields.

        Args:
            config: Parsed configuration dictionary.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[ValidationError] = []

        # Check for required top-level sections
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                errors.append(
                    ValidationError(
                        message=f"Missing required section: '{section}'",
                        field=section,
                        suggestions=[
                            f"Add '{section}:' section to your configuration",
                            "Check examples in configs/ directory",
                        ],
                    )
                )

        # Validate individual sections
        if "model" in config:
            errors.extend(self._validate_model_section(config["model"]))

        if "training" in config:
            errors.extend(self._validate_training_section(config["training"]))

        if "data" in config:
            errors.extend(self._validate_data_section(config["data"]))

        return len(errors) == 0, errors

    def validate_types(
        self, config: dict[str, object]
    ) -> tuple[bool, list[ValidationError]]:
        """Validate data types in configuration.

        Args:
            config: Parsed configuration dictionary.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[ValidationError] = []

        # Define expected types for common fields
        type_expectations = {
            "training.epochs": int,
            "training.batch_size": int,
            "training.learning_rate": (int, float),
            "model.num_classes": int,
            "model.input_channels": int,
            "data.split_ratio": (list, tuple),
            "data.augment": bool,
        }

        for field_path, expected_type in type_expectations.items():
            value = self._get_nested_value(config, field_path)
            if value is not None and not isinstance(value, expected_type):
                type_name = (
                    " or ".join(t.__name__ for t in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                errors.append(
                    ValidationError(
                        message=(
                            f"Invalid type for {field_path}: expected "
                            f"{type_name}, got {type(value).__name__}"
                        ),
                        field=field_path,
                        suggestions=[
                            f"Change {field_path} to {type_name} type",
                            (
                                f"Example: {field_path}: "
                                f"{self._get_type_example(expected_type)}"
                            ),
                        ],
                    )
                )

        return len(errors) == 0, errors

    def validate_values(
        self, config: dict[str, object]
    ) -> tuple[bool, list[ValidationError]]:
        """Validate specific value constraints.

        Args:
            config: Parsed configuration dictionary.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[ValidationError] = []

        # Validate model architecture
        model_arch = self._get_nested_value(config, "model.architecture")
        if model_arch and model_arch not in self.model_architectures:
            errors.append(
                ValidationError(
                    message=f"Unknown model architecture: '{model_arch}'",
                    field="model.architecture",
                    suggestions=[
                        (
                            f"Use one of: "
                            f"{', '.join(sorted(self.model_architectures))}"
                        ),
                        "Check available architectures in src/model/",
                    ],
                )
            )

        # Validate encoder
        encoder = self._get_nested_value(config, "model.encoder.type")
        if encoder and encoder not in self.encoders:
            errors.append(
                ValidationError(
                    message=f"Unknown encoder type: '{encoder}'",
                    field="model.encoder.type",
                    suggestions=[
                        f"Use one of: {', '.join(sorted(self.encoders))}",
                        "Check available encoders in model documentation",
                    ],
                )
            )

        # Validate loss function
        loss_fn = self._get_nested_value(config, "training.loss.type")
        if loss_fn and loss_fn not in self.loss_functions:
            errors.append(
                ValidationError(
                    message=f"Unknown loss function: '{loss_fn}'",
                    field="training.loss.type",
                    suggestions=[
                        (
                            f"Use one of: "
                            f"{', '.join(sorted(self.loss_functions))}"
                        ),
                        (
                            "Check available loss functions in "
                            "src/training/losses/"
                        ),
                    ],
                )
            )

        # Validate numeric ranges
        epochs = self._get_nested_value(config, "training.epochs")
        if epochs is not None and isinstance(epochs, int) and epochs <= 0:
            errors.append(
                ValidationError(
                    message=(
                        f"Invalid epochs value: {epochs} (must be positive)"
                    ),
                    field="training.epochs",
                    suggestions=["Use a positive integer (e.g., epochs: 100)"],
                )
            )

        batch_size = self._get_nested_value(config, "training.batch_size")
        if (
            batch_size is not None
            and isinstance(batch_size, int)
            and batch_size <= 0
        ):
            errors.append(
                ValidationError(
                    message=(
                        f"Invalid batch_size value: {batch_size} "
                        "(must be positive)"
                    ),
                    field="training.batch_size",
                    suggestions=[
                        "Use a positive integer (e.g., batch_size: 16)"
                    ],
                )
            )

        return len(errors) == 0, errors

    def comprehensive_validate(
        self, content: str
    ) -> tuple[bool, list[ValidationError]]:
        """Perform comprehensive validation of YAML content.

        Args:
            content: YAML content as string.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        all_errors: list[ValidationError] = []

        # Step 1: Syntax validation
        syntax_valid, syntax_error = self.validate_syntax(content)
        if not syntax_valid:
            return False, [syntax_error] if syntax_error else []

        # Step 2: Parse and validate structure
        try:
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                return False, [
                    ValidationError(
                        message="Configuration must be a YAML dictionary",
                        suggestions=[
                            (
                                "Ensure your configuration starts with key: "
                                "value pairs"
                            )
                        ],
                    )
                ]
        except Exception as e:
            return False, [
                ValidationError(message=f"Failed to parse YAML: {str(e)}")
            ]

        # Step 3: Structure validation
        structure_valid, structure_errors = self.validate_structure(config)
        all_errors.extend(structure_errors)

        # Step 4: Type validation
        types_valid, type_errors = self.validate_types(config)
        all_errors.extend(type_errors)

        # Step 5: Value validation
        values_valid, value_errors = self.validate_values(config)
        all_errors.extend(value_errors)

        return len(all_errors) == 0, all_errors

    def _validate_model_section(
        self, model_config: object
    ) -> list[ValidationError]:
        """Validate model configuration section."""
        errors: list[ValidationError] = []

        if not isinstance(model_config, dict):
            errors.append(
                ValidationError(
                    message="Model section must be a dictionary",
                    field="model",
                    suggestions=[
                        "Use 'model:' followed by indented key-value pairs"
                    ],
                )
            )
            return errors

        # Check required model fields
        required_model_fields = ["architecture"]
        for field in required_model_fields:
            if field not in model_config:
                errors.append(
                    ValidationError(
                        message=f"Missing required model field: '{field}'",
                        field=f"model.{field}",
                        suggestions=[
                            f"Add 'model.{field}:' to your configuration",
                            f"Example: {field}: unet",
                        ],
                    )
                )

        return errors

    def _validate_training_section(
        self, training_config: object
    ) -> list[ValidationError]:
        """Validate training configuration section."""
        errors: list[ValidationError] = []

        if not isinstance(training_config, dict):
            errors.append(
                ValidationError(
                    message="Training section must be a dictionary",
                    field="training",
                    suggestions=[
                        "Use 'training:' followed by indented key-value pairs"
                    ],
                )
            )
            return errors

        return errors

    def _validate_data_section(
        self, data_config: object
    ) -> list[ValidationError]:
        """Validate data configuration section."""
        errors: list[ValidationError] = []

        if not isinstance(data_config, dict):
            errors.append(
                ValidationError(
                    message="Data section must be a dictionary",
                    field="data",
                    suggestions=[
                        "Use 'data:' followed by indented key-value pairs"
                    ],
                )
            )
            return errors

        return errors

    def _get_nested_value(
        self, config: dict[str, object], path: str
    ) -> object | None:
        """Get value from nested dictionary using dot notation."""
        keys = path.split(".")
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _get_syntax_suggestions(
        self, error_message: str, content: str, line_num: int | None
    ) -> list[str]:
        """Generate syntax error suggestions based on common patterns."""
        suggestions = []

        if "could not find expected ':'" in error_message.lower():
            suggestions.extend(
                [
                    "Add missing colon (:) after the key name",
                    "Check for proper indentation",
                    "Ensure key-value pairs are properly formatted",
                ]
            )

        if "found unexpected end of stream" in error_message.lower():
            suggestions.extend(
                [
                    "Check for missing closing brackets or quotes",
                    "Ensure the file is not truncated",
                    "Verify all indented blocks are complete",
                ]
            )

        if "indentation" in error_message.lower():
            suggestions.extend(
                [
                    "Use consistent indentation (spaces or tabs, not mixed)",
                    "Ensure child elements are indented more than parent",
                    "Check for trailing spaces or tabs",
                ]
            )

        return suggestions

    def _get_type_example(self, expected_type: type | tuple[type, ...]) -> str:
        """Get example value for a given type."""
        if expected_type is int:
            return "42"
        elif expected_type is float or expected_type == (int, float):
            return "0.001"
        elif expected_type is bool:
            return "true"
        elif expected_type is str:
            return "'example_string'"
        elif expected_type is list or expected_type == (list, tuple):
            return "[0.8, 0.1, 0.1]"
        else:
            return "value"
