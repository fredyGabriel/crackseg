"""
Checkpoint integrity verification.

This module provides specialized integrity verification for PyTorch
checkpoints, extending the existing checkpoint verification functionality.
"""

import logging
from pathlib import Path

import torch

from .core import IntegrityVerifier, VerificationLevel, VerificationResult

logger = logging.getLogger(__name__)


class CheckpointIntegrityVerifier(IntegrityVerifier):
    """Specialized verifier for PyTorch checkpoint integrity."""

    def __init__(
        self,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
        required_fields: list[str] | None = None,
    ):
        """
        Initialize checkpoint integrity verifier.

        Args:
            verification_level: Level of verification thoroughness
            required_fields: List of required fields in checkpoint
        """
        super().__init__(verification_level)
        self.required_fields = required_fields or [
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "pytorch_version",
            "timestamp",
        ]

    def verify(self, artifact_path: Path) -> VerificationResult:
        """
        Verify the integrity of a PyTorch checkpoint.

        Args:
            artifact_path: Path to the checkpoint file

        Returns:
            VerificationResult with verification details
        """
        result = VerificationResult(
            is_valid=True,
            artifact_path=artifact_path,
            verification_level=self.verification_level,
        )

        # Basic structure verification
        if not self._verify_basic_structure(artifact_path, result):
            return result

        # Calculate checksum
        checksum = self._verify_checksum(artifact_path)
        result.checksum = checksum

        if self.verification_level in [
            VerificationLevel.STANDARD,
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Load and validate checkpoint content
            if not self._verify_checkpoint_content(artifact_path, result):
                return result

        if self.verification_level in [
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Deep content analysis
            self._verify_model_structure(artifact_path, result)

        if self.verification_level == VerificationLevel.PARANOID:
            # Cross-reference validation
            self._verify_cross_references(artifact_path, result)

        return result

    def _verify_checkpoint_content(
        self, checkpoint_path: Path, result: VerificationResult
    ) -> bool:
        """Verify checkpoint content and required fields."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                # Explicitly allow non-model data for compatibility
                weights_only=False,
            )

            # Verify required fields
            missing_fields = []
            for field in self.required_fields:
                if field not in checkpoint_data:
                    missing_fields.append(field)

            if missing_fields:
                result.add_error(f"Missing required fields: {missing_fields}")
                return False

            # Add metadata from checkpoint
            result.add_metadata("epoch", checkpoint_data.get("epoch"))
            result.add_metadata(
                "pytorch_version", checkpoint_data.get("pytorch_version")
            )
            result.add_metadata("timestamp", checkpoint_data.get("timestamp"))
            result.add_metadata(
                "model_keys",
                list(checkpoint_data.get("model_state_dict", {}).keys()),
            )

            # Verify model state dict structure
            model_state_dict = checkpoint_data.get("model_state_dict", {})
            if not isinstance(model_state_dict, dict):
                result.add_error("model_state_dict is not a dictionary")
                return False

            if not model_state_dict:
                result.add_warning("model_state_dict is empty")

            # Verify optimizer state dict structure
            optimizer_state_dict = checkpoint_data.get(
                "optimizer_state_dict", {}
            )
            if not isinstance(optimizer_state_dict, dict):
                result.add_error("optimizer_state_dict is not a dictionary")
                return False

            return True

        except Exception as e:
            result.add_error(f"Failed to load checkpoint: {e}")
            return False

    def _verify_model_structure(
        self, checkpoint_path: Path, result: VerificationResult
    ) -> None:
        """Verify model structure and tensor properties."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                # Explicitly allow non-model data for compatibility
                weights_only=False,
            )

            model_state_dict = checkpoint_data.get("model_state_dict", {})

            # Analyze tensor properties
            tensor_count = 0
            total_params = 0
            param_shapes = {}

            for key, tensor in model_state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_count += 1
                    total_params += tensor.numel()
                    param_shapes[key] = list(tensor.shape)

                    # Check for NaN or Inf values
                    if torch.isnan(tensor).any():
                        result.add_warning(
                            f"NaN values detected in parameter: {key}"
                        )

                    if torch.isinf(tensor).any():
                        result.add_warning(
                            f"Inf values detected in parameter: {key}"
                        )

            result.add_metadata("tensor_count", tensor_count)
            result.add_metadata("total_parameters", total_params)
            result.add_metadata("parameter_shapes", param_shapes)

        except Exception as e:
            result.add_warning(f"Deep structure analysis failed: {e}")

    def _verify_cross_references(
        self, checkpoint_path: Path, result: VerificationResult
    ) -> None:
        """Verify cross-references and consistency checks."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                # Explicitly allow non-model data for compatibility
                weights_only=False,
            )

            # Verify optimizer state matches model state
            model_state_dict = checkpoint_data.get("model_state_dict", {})
            optimizer_state_dict = checkpoint_data.get(
                "optimizer_state_dict", {}
            )

            if "state" in optimizer_state_dict:
                optimizer_state = optimizer_state_dict["state"]
                model_keys = set(model_state_dict.keys())

                # Check if optimizer state references match model parameters
                for param_id, _param_state in optimizer_state.items():
                    if "param_groups" in optimizer_state_dict:
                        param_groups = optimizer_state_dict["param_groups"]
                        if param_id < len(param_groups):
                            param_group = param_groups[param_id]
                            if "params" in param_group:
                                for param_idx in param_group["params"]:
                                    if param_idx < len(model_keys):
                                        model_key = list(model_keys)[param_idx]
                                        if model_key not in model_state_dict:
                                            result.add_warning(
                                                "Optimizer references missing "
                                                f"model parameter: {model_key}"
                                            )

            # Verify PyTorch version compatibility
            pytorch_version = checkpoint_data.get("pytorch_version")
            current_version = torch.__version__

            if pytorch_version and pytorch_version != current_version:
                result.add_warning(
                    f"Checkpoint created with PyTorch {pytorch_version}, "
                    f"current version is {current_version}"
                )

        except Exception as e:
            result.add_warning(f"Cross-reference verification failed: {e}")
