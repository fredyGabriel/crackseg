"""Checkpoint validation functionality.

This module handles checkpoint validation and integrity verification.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from .config import CheckpointSpec

logger = logging.getLogger(__name__)


def verify_checkpoint_integrity(
    checkpoint_path: str | Path, spec: CheckpointSpec | None = None
) -> dict[str, Any]:
    """Verify checkpoint integrity and completeness.

    Args:
        checkpoint_path: Path to checkpoint file
        spec: Specification for validation

    Returns:
        Dictionary with verification results
    """
    if spec is None:
        spec = CheckpointSpec()

    checkpoint_path = Path(checkpoint_path)

    # Initialize validation variables
    is_valid = False
    missing_fields: list[str] = []

    verification_result: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "is_valid": False,
        "missing_fields": [],
        "size_bytes": 0,
        "error": None,
    }

    if not checkpoint_path.exists():
        verification_result["error"] = "File does not exist"
        return verification_result

    try:
        # Get file size
        verification_result["size_bytes"] = checkpoint_path.stat().st_size

        # Load and validate checkpoint
        checkpoint_data = torch.load(
            checkpoint_path, map_location=torch.device("cpu")
        )

        # Validate completeness
        is_valid, missing_fields = validate_checkpoint_completeness(
            checkpoint_data, spec
        )

        verification_result["is_valid"] = is_valid
        verification_result["missing_fields"] = missing_fields

        # Add metadata from checkpoint
        verification_result["epoch"] = checkpoint_data.get("epoch")
        verification_result["pytorch_version"] = checkpoint_data.get(
            "pytorch_version"
        )
        verification_result["timestamp"] = checkpoint_data.get("timestamp")

        logger.info(f"Checkpoint verification completed for {checkpoint_path}")

    except Exception as e:
        verification_result["error"] = f"Failed to load/verify checkpoint: {e}"
        logger.error(
            f"Checkpoint verification failed for {checkpoint_path}: {e}"
        )

    if not is_valid:
        logger.error(
            f"Checkpoint validation failed. Missing fields: {missing_fields}"
        )
        raise ValueError(
            f"Invalid checkpoint format. Missing required fields: "
            f"{missing_fields}"
        )

    return verification_result


def validate_checkpoint_completeness(
    checkpoint_data: dict[str, Any], spec: CheckpointSpec | None = None
) -> tuple[bool, list[str]]:
    """Validate that checkpoint contains all required fields.

    Args:
        checkpoint_data: Checkpoint dictionary to validate
        spec: Specification defining required fields

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    if spec is None:
        spec = CheckpointSpec()

    missing_fields = []
    for field in spec.required_fields:
        if field not in checkpoint_data:
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields


def validate_checkpoint_format(checkpoint_data: dict[str, Any]) -> bool:
    """Validate basic checkpoint format without strict requirements.

    Args:
        checkpoint_data: Checkpoint dictionary to validate

    Returns:
        True if checkpoint has basic required structure
    """
    # Basic validation - just check for essential fields
    essential_fields = {"model_state_dict"}

    for field in essential_fields:
        if field not in checkpoint_data:
            logger.warning(f"Checkpoint missing essential field: {field}")
            return False

    return True


def get_checkpoint_metadata(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from checkpoint data.

    Args:
        checkpoint_data: Checkpoint dictionary

    Returns:
        Dictionary containing checkpoint metadata
    """
    metadata_fields = {
        "epoch",
        "pytorch_version",
        "python_version",
        "timestamp",
        "experiment_id",
        "git_commit",
        "notes",
        "best_metric_value",
    }

    metadata = {}
    for field in metadata_fields:
        if field in checkpoint_data:
            metadata[field] = checkpoint_data[field]

    return metadata
