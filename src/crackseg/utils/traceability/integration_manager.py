"""
Integration manager for metadata and access control in CrackSeg project.

This module provides integrated functionality combining metadata management
and access control capabilities for comprehensive artifact traceability.
"""

import logging
from typing import Any

from .access_control import AccessControl
from .integration_ops import (
    append_bulk_result,
)
from .integration_ops import (
    audit_trace_with_access_control as _audit_trace_with_access_control,
)
from .integration_ops import (
    build_bulk_results_header as _build_bulk_results_header,
)
from .integration_ops import (
    get_metadata_statistics_with_access as _get_metadata_statistics_with_access,
)
from .integration_ops import (
    search_with_access_control as _search_with_access_control,
)
from .integration_ops import (
    validate_compliance_with_access as _validate_compliance_with_access,
)

# keep compatibility imports local where used; avoid redundant globals
from .metadata_manager import MetadataManager
from .storage import TraceabilityStorage
from .utils.integration import build_access_log_entry, build_success_response

logger = logging.getLogger(__name__)


class TraceabilityIntegrationManager:
    """Integrated manager for metadata and access control."""

    def __init__(self, storage: TraceabilityStorage) -> None:
        """Initialize integration manager with storage.

        Args:
            storage: Traceability storage instance
        """
        self.storage = storage
        self.metadata_manager = MetadataManager(storage)
        self.access_control = AccessControl(storage)

    def enrich_artifact_with_access_control(
        self,
        artifact_id: str,
        user_id: str,
        metadata: dict[str, Any],
        permission: str = "write",
    ) -> dict[str, Any]:
        """Enrich artifact with metadata while checking access control.

        Args:
            artifact_id: Artifact ID to enrich
            user_id: User ID requesting enrichment
            metadata: Additional metadata to add
            permission: Permission required for enrichment

        Returns:
            Dictionary with enrichment results and access control info

        Raises:
            RuntimeError: If access is denied or enrichment fails
        """
        # Check access control first
        has_access = self.access_control.check_artifact_access(
            artifact_id, user_id, permission
        )

        if not has_access:
            raise RuntimeError(
                f"Access denied: User {user_id} cannot {permission} "
                f"artifact {artifact_id}"
            )

        # Get artifact from storage
        artifacts_data = self.storage._load_artifacts()
        artifact = None

        for a in artifacts_data:
            if a.get("artifact_id") == artifact_id:
                artifact = a
                break

        if not artifact:
            raise RuntimeError(f"Artifact {artifact_id} not found")

        # Convert dictionary to ArtifactEntity
        from .entities import ArtifactEntity

        artifact_entity = ArtifactEntity(**artifact)

        # Enrich with metadata
        try:
            enriched_artifact = self.metadata_manager.enrich_artifact_metadata(
                artifact_entity, metadata
            )

            # Log access control event
            access_log = self.access_control.get_access_log(user_id)
            access_log.append(
                build_access_log_entry(
                    user_id,
                    "artifact",
                    artifact_id,
                    f"enrich_metadata_{permission}",
                    "granted",
                )
            )

            return build_success_response(
                {
                    "artifact_id": artifact_id,
                    "enriched_artifact": enriched_artifact,
                    "access_granted": True,
                    "metadata_added": list(metadata.keys()),
                }
            )

        except Exception as e:
            logger.error(f"Failed to enrich artifact {artifact_id}: {e}")
            raise RuntimeError(f"Enrichment failed: {e}") from e

    def search_with_access_control(
        self,
        metadata_key: str,
        metadata_value: Any,
        user_id: str,
        entity_type: str = "artifact",
    ) -> dict[str, Any]:
        """Search entities by metadata with access control filtering.

        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match
            user_id: User ID performing search
            entity_type: Type of entity to search

        Returns:
            Dictionary with search results and access control info
        """
        return _search_with_access_control(
            self.access_control,
            self.metadata_manager,
            metadata_key,
            metadata_value,
            user_id,
            entity_type,
        )

    def get_metadata_statistics_with_access(
        self, user_id: str
    ) -> dict[str, Any]:
        """Get metadata statistics filtered by user access.

        Args:
            user_id: User ID to get statistics for

        Returns:
            Dictionary with metadata statistics and access control info
        """
        return _get_metadata_statistics_with_access(
            self.access_control, self.metadata_manager, user_id
        )

    def validate_compliance_with_access(
        self, entity_type: str, entity_id: str, user_id: str
    ) -> dict[str, Any]:
        """Validate compliance with access control checks.

        Args:
            entity_type: Type of entity to validate
            entity_id: Entity ID to validate
            user_id: User ID requesting validation

        Returns:
            Dictionary with compliance validation and access control info
        """
        return _validate_compliance_with_access(
            self.access_control,
            self.metadata_manager,
            entity_type,
            entity_id,
            user_id,
        )

    def audit_trace_with_access_control(
        self, user_id: str, entity_type: str | None = None
    ) -> dict[str, Any]:
        """Get audit trail with access control filtering.

        Args:
            user_id: User ID to get audit trail for
            entity_type: Optional entity type filter

        Returns:
            Dictionary with audit trail and access control info
        """
        return _audit_trace_with_access_control(
            self.access_control, user_id, entity_type
        )

    def bulk_metadata_operation_with_access(
        self,
        operation: str,
        entity_ids: list[str],
        user_id: str,
        metadata: dict[str, Any],
        entity_type: str = "artifact",
    ) -> dict[str, Any]:
        """Perform bulk metadata operations with access control.

        Args:
            operation: Operation type ("enrich", "validate", "search")
            entity_ids: List of entity IDs to operate on
            user_id: User ID performing operation
            metadata: Metadata for operation
            entity_type: Type of entities

        Returns:
            Dictionary with bulk operation results
        """
        results = _build_bulk_results_header(
            operation=operation,
            user_id=user_id,
            entity_type=entity_type,
            total=len(entity_ids),
        )

        for entity_id in entity_ids:
            try:
                # Check access control
                has_access = False
                if entity_type == "artifact":
                    has_access = self.access_control.check_artifact_access(
                        entity_id, user_id, "write"
                    )
                elif entity_type == "experiment":
                    has_access = self.access_control.check_experiment_access(
                        entity_id, user_id, "write"
                    )

                if not has_access:
                    append_bulk_result(results, entity_id, "access_denied")
                    continue

                # Perform operation
                if operation == "enrich":
                    operation_result = (
                        self.enrich_artifact_with_access_control(
                            entity_id, user_id, metadata
                        )
                    )
                elif operation == "validate":
                    operation_result = self.validate_compliance_with_access(
                        entity_type, entity_id, user_id
                    )
                else:
                    operation_result = {
                        "error": f"Unknown operation: {operation}"
                    }

                status = (
                    "success"
                    if operation_result.get("success", False)
                    else "failed"
                )
                append_bulk_result(
                    results, entity_id, status, operation_result
                )

            except Exception as e:
                append_bulk_result(
                    results, entity_id, "error", {"error": str(e)}
                )

        return results
