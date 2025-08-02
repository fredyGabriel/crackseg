"""
Access control system for CrackSeg project.

This module provides comprehensive access control capabilities for the
traceability system, including user permissions, role-based access,
and access validation.
"""

import logging
from datetime import datetime
from typing import Any

from .storage import TraceabilityStorage

logger = logging.getLogger(__name__)


class AccessControl:
    """Access control system for traceability entities."""

    def __init__(self, storage: TraceabilityStorage) -> None:
        """Initialize access control with storage.

        Args:
            storage: Traceability storage instance
        """
        self.storage = storage

    def check_artifact_access(
        self, artifact_id: str, user_id: str, permission: str
    ) -> bool:
        """Check if user has permission to access artifact.

        Args:
            artifact_id: Artifact ID to check access for
            user_id: User ID requesting access
            permission: Permission type ("read", "write", "delete")

        Returns:
            True if access is granted, False otherwise
        """
        # Admin users have access to all artifacts
        if user_id == "admin_user":
            return True

        # Get artifact
        artifacts_data = self.storage._load_artifacts()
        artifact = None

        for a in artifacts_data:
            if a.get("artifact_id") == artifact_id:
                artifact = a
                break

        if not artifact:
            return False  # Artifact doesn't exist

        # Check ownership
        owner = artifact.get("owner", "")
        if owner == user_id:
            return True  # Owner has all permissions

        # Check public access for read
        if permission == "read":
            compliance_level = artifact.get("compliance_level", "basic")
            if compliance_level in ["public", "standard"]:
                return True

        # Check specific permissions based on artifact type
        artifact_type = artifact.get("artifact_type", "")
        if self._check_type_based_permission(
            artifact_type, user_id, permission
        ):
            return True

        return False

    def check_experiment_access(
        self, experiment_id: str, user_id: str, permission: str
    ) -> bool:
        """Check if user has permission to access experiment.

        Args:
            experiment_id: Experiment ID to check access for
            user_id: User ID requesting access
            permission: Permission type ("read", "write", "delete")

        Returns:
            True if access is granted, False otherwise
        """
        # Get experiment
        experiments_data = self.storage._load_experiments()
        experiment = None

        for e in experiments_data:
            if e.get("experiment_id") == experiment_id:
                experiment = e
                break

        if not experiment:
            return False  # Experiment doesn't exist

        # Check ownership (experiments don't have explicit owner, use username)
        username = experiment.get("username", "")
        if username == user_id:
            return True  # Owner has all permissions

        # Check public access for read
        if permission == "read":
            status = experiment.get("status", "created")
            if status in ["completed", "published"]:
                return True

        return False

    def get_user_permissions(self, user_id: str) -> dict[str, Any]:
        """Get comprehensive permissions for a user.

        Args:
            user_id: User ID to get permissions for

        Returns:
            Dictionary with user permissions
        """
        artifacts_data = self.storage._load_artifacts()
        experiments_data = self.storage._load_experiments()

        # Count owned artifacts
        owned_artifacts = [
            a for a in artifacts_data if a.get("owner") == user_id
        ]

        # Count owned experiments
        owned_experiments = [
            e for e in experiments_data if e.get("username") == user_id
        ]

        # Count accessible artifacts (owned + public)
        accessible_artifacts = [
            a
            for a in artifacts_data
            if (
                a.get("owner") == user_id
                or a.get("compliance_level") in ["public", "standard"]
            )
        ]

        # Count accessible experiments (owned + completed)
        accessible_experiments = [
            e
            for e in experiments_data
            if (
                e.get("username") == user_id
                or e.get("status") in ["completed", "published"]
            )
        ]

        return {
            "user_id": user_id,
            "owned_artifacts": len(owned_artifacts),
            "owned_experiments": len(owned_experiments),
            "accessible_artifacts": len(accessible_artifacts),
            "accessible_experiments": len(accessible_experiments),
            "can_create_artifacts": True,  # All users can create
            "can_create_experiments": True,  # All users can create
            "can_delete_own_artifacts": True,
            "can_delete_own_experiments": True,
            "can_access_public_data": True,
        }

    def validate_access_request(
        self, entity_type: str, entity_id: str, user_id: str, action: str
    ) -> dict[str, Any]:
        """Validate an access request and return detailed results.

        Args:
            entity_type: Type of entity ("artifact", "experiment")
            entity_id: Entity ID to access
            user_id: User ID requesting access
            action: Action being performed ("read", "write", "delete")

        Returns:
            Dictionary with validation results
        """
        if entity_type == "artifact":
            has_access = self.check_artifact_access(entity_id, user_id, action)
        elif entity_type == "experiment":
            has_access = self.check_experiment_access(
                entity_id, user_id, action
            )
        else:
            return {
                "valid": False,
                "error": f"Unsupported entity type: {entity_type}",
                "user_id": user_id,
                "entity_id": entity_id,
                "action": action,
            }

        return {
            "valid": has_access,
            "user_id": user_id,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "error": None if has_access else "Access denied",
        }

    def get_access_log(
        self, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get access control log (simulated for now).

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of access log entries
        """
        # This would typically be stored in a separate log file
        # For now, we'll return a simulated log
        log_entries = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id or "system",
                "entity_type": "artifact",
                "entity_id": "artifact-001",
                "action": "read",
                "result": "granted",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id or "system",
                "entity_type": "experiment",
                "entity_id": "exp-001",
                "action": "write",
                "result": "granted",
            },
        ]

        if user_id:
            log_entries = [
                entry for entry in log_entries if entry["user_id"] == user_id
            ]

        return log_entries

    def _check_type_based_permission(
        self, artifact_type: str, user_id: str, permission: str
    ) -> bool:
        """Check permissions based on artifact type.

        Args:
            artifact_type: Type of artifact
            user_id: User ID requesting access
            permission: Permission type

        Returns:
            True if permission granted, False otherwise
        """
        # For now, only allow access to owners
        # In a real system, you'd check user roles here
        return False  # Default to no access for non-owners

    def enforce_compliance_policy(
        self, entity_type: str, entity_id: str
    ) -> dict[str, Any]:
        """Enforce compliance policies on entities.

        Args:
            entity_type: Type of entity ("artifact", "experiment")
            entity_id: Entity ID to check

        Returns:
            Compliance enforcement results
        """
        if entity_type == "artifact":
            return self._enforce_artifact_compliance(entity_id)
        elif entity_type == "experiment":
            return self._enforce_experiment_compliance(entity_id)
        else:
            return {
                "valid": False,
                "error": f"Unsupported entity type: {entity_type}",
            }

    def _enforce_artifact_compliance(self, artifact_id: str) -> dict[str, Any]:
        """Enforce compliance policies on artifacts.

        Args:
            artifact_id: Artifact ID to check

        Returns:
            Compliance enforcement results
        """
        artifacts_data = self.storage._load_artifacts()
        artifact = None

        for a in artifacts_data:
            if a.get("artifact_id") == artifact_id:
                artifact = a
                break

        if not artifact:
            return {
                "valid": False,
                "error": "Artifact not found",
            }

        # Check compliance requirements
        compliance_level = artifact.get("compliance_level", "basic")
        verification_status = artifact.get("verification_status", "pending")

        issues = []

        # Check verification status
        if verification_status != "verified":
            issues.append("Artifact not verified")

        # Check compliance level requirements
        if compliance_level == "standard":
            if not artifact.get("checksum"):
                issues.append("Standard compliance requires checksum")
            if not artifact.get("description"):
                issues.append("Standard compliance requires description")

        elif compliance_level == "strict":
            if not artifact.get("checksum"):
                issues.append("Strict compliance requires checksum")
            if not artifact.get("description"):
                issues.append("Strict compliance requires description")
            if not artifact.get("metadata", {}).get("accuracy"):
                issues.append("Strict compliance requires accuracy metadata")

        return {
            "valid": len(issues) == 0,
            "compliance_level": compliance_level,
            "verification_status": verification_status,
            "issues": issues,
            "artifact_id": artifact_id,
        }

    def _enforce_experiment_compliance(
        self, experiment_id: str
    ) -> dict[str, Any]:
        """Enforce compliance policies on experiments.

        Args:
            experiment_id: Experiment ID to check

        Returns:
            Compliance enforcement results
        """
        experiments_data = self.storage._load_experiments()
        experiment = None

        for e in experiments_data:
            if e.get("experiment_id") == experiment_id:
                experiment = e
                break

        if not experiment:
            return {
                "valid": False,
                "error": "Experiment not found",
            }

        # Check compliance requirements
        status = experiment.get("status", "created")
        issues = []

        # Check required fields for completed experiments
        if status == "completed":
            if not experiment.get("description"):
                issues.append("Completed experiments require description")
            if not experiment.get("best_metrics"):
                issues.append("Completed experiments require metrics")
            if not experiment.get("config_summary"):
                issues.append("Completed experiments require config summary")

        return {
            "valid": len(issues) == 0,
            "status": status,
            "issues": issues,
            "experiment_id": experiment_id,
        }
