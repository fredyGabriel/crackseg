"""Custom exceptions for the crack segmentation project."""


class CrackSegError(Exception):
    """Base exception class for the crack segmentation project."""

    def __init__(self, message: str, details: str | None = None):
        """Initialize the exception.

        Args:
            message: Main error message
            details: Optional additional error details
        """
        self.message: str = message
        self.details: str | None = details
        super().__init__(self.full_message)

    @property
    def full_message(self) -> str:
        """Get the complete error message including details if available."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigError(CrackSegError):
    """Raised when there is an error in the configuration."""

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        field: str | None = None,
        details: str | None = None,
    ):
        """Initialize the configuration error.

        Args:
            message: Main error message
            config_path: Path to the problematic config file/section
            field: Name of the field that failed validation (optional)
            details: Optional additional error details
        """
        self.config_path: str | None = config_path
        self.field: str | None = field
        if config_path:
            message = f"Configuration error in {config_path}: {message}"
        if field:
            message = f"Config error for {field}: {message}"
        super().__init__(message, details)


class DataError(CrackSegError):
    """Raised when there is an error with the input data."""

    def __init__(
        self,
        message: str,
        data_path: str | None = None,
        details: str | None = None,
    ):
        """Initialize the data error.

        Args:
            message: Main error message
            data_path: Path to the problematic data file/directory
            details: Optional additional error details
        """
        self.data_path: str | None = data_path
        if data_path:
            message = f"Data error in {data_path}: {message}"
        super().__init__(message, details)


class ModelError(CrackSegError):
    """Raised when there is an error with the model."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        details: str | None = None,
    ):
        """Initialize the model error.

        Args:
            message: Main error message
            model_name: Name or identifier of the problematic model
            details: Optional additional error details
        """
        self.model_name: str | None = model_name
        if model_name:
            message = f"Model error in {model_name}: {message}"
        super().__init__(message, details)


class TrainingError(CrackSegError):
    """Raised when there is an error during model training."""

    def __init__(
        self,
        message: str,
        epoch: int | None = None,
        details: str | None = None,
    ):
        """Initialize the training error.

        Args:
            message: Main error message
            epoch: Training epoch where the error occurred
            details: Optional additional error details
        """
        self.epoch: int | None = epoch
        if epoch is not None:
            message = f"Training error at epoch {epoch}: {message}"
        super().__init__(message, details)


class EvaluationError(CrackSegError):
    """Raised when there is an error during model evaluation."""

    def __init__(
        self,
        message: str,
        checkpoint: str | None = None,
        details: str | None = None,
    ):
        """Initialize the evaluation error.

        Args:
            message: Main error message
            checkpoint: Path or name of the checkpoint being evaluated
            details: Optional additional error details
        """
        self.checkpoint: str | None = checkpoint
        if checkpoint:
            message = f"Evaluation error with checkpoint {checkpoint}: \
{message}"
        super().__init__(message, details)


class ResourceError(CrackSegError):
    """Raised when there is an error with system resources."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        details: str | None = None,
    ):
        """Initialize the resource error.

        Args:
            message: Main error message
            resource_type: Type of resource (e.g., 'GPU', 'Memory')
            details: Optional additional error details
        """
        self.resource_type: str | None = resource_type
        if resource_type:
            message = f"{resource_type} resource error: {message}"
        super().__init__(message, details)


class ValidationError(CrackSegError):
    """Raised when there is an error during data or config validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: str | None = None,
    ):
        """Initialize the validation error.

        Args:
            message: Main error message
            field: Name of the field that failed validation
            details: Optional additional error details
        """
        self.field: str | None = field
        if field:
            message = f"Validation error for {field}: {message}"
        super().__init__(message, details)
