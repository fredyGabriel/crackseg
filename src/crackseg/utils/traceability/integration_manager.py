"""
Integration manager for metadata and access control in CrackSeg project.

This module provides integrated functionality combining metadata management
and access control capabilities for comprehensive artifact traceability.
"""

import logging
from datetime import datetime
from typing import Any

from .access_control import AccessControl
from .metadata_manager import MetadataManager
from .storage import TraceabilityStorage

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
                {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "entity_type": "artifact",
                    "entity_id": artifact_id,
                    "action": f"enrich_metadata_{permission}",
                    "result": "granted",
                }
            )

            return {
                "success": True,
                "artifact_id": artifact_id,
                "enriched_artifact": enriched_artifact,
                "access_granted": True,
                "metadata_added": list(metadata.keys()),
                "timestamp": datetime.now().isoformat(),
            }

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
        # Get all matching entities
        matching_entities = self.metadata_manager.search_by_metadata(
            metadata_key, metadata_value, entity_type
        )

        # Filter by access control
        accessible_entities = []

        for entity in matching_entities:
            entity_id = entity.get(f"{entity_type}_id", "")

            if entity_type == "artifact":
                has_access = self.access_control.check_artifact_access(
                    entity_id, user_id, "read"
                )
            elif entity_type == "experiment":
                has_access = self.access_control.check_experiment_access(
                    entity_id, user_id, "read"
                )
            else:
                has_access = True  # Default for other entity types

            if has_access:
                accessible_entities.append(entity)

        return {
            "success": True,
            "user_id": user_id,
            "metadata_key": metadata_key,
            "metadata_value": metadata_value,
            "entity_type": entity_type,
            "total_matches": len(matching_entities),
            "accessible_matches": len(accessible_entities),
            "accessible_entities": accessible_entities,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metadata_statistics_with_access(
        self, user_id: str
    ) -> dict[str, Any]:
        """Get metadata statistics filtered by user access.

        Args:
            user_id: User ID to get statistics for

        Returns:
            Dictionary with metadata statistics and access control info
        """
        # Get user permissions
        user_permissions = self.access_control.get_user_permissions(user_id)

        # Get metadata statistics
        metadata_stats = self.metadata_manager.get_metadata_statistics()

        # Filter statistics based on access
        accessible_stats = {
            "user_id": user_id,
            "total_accessible_artifacts": user_permissions[
                "accessible_artifacts"
            ],
            "total_accessible_experiments": user_permissions[
                "accessible_experiments"
            ],
            "owned_artifacts": user_permissions["owned_artifacts"],
            "owned_experiments": user_permissions["owned_experiments"],
            "metadata_keys_accessible": metadata_stats.get(
                "artifact_metadata_keys", []
            ),
            "can_create_artifacts": user_permissions["can_create_artifacts"],
            "can_create_experiments": user_permissions[
                "can_create_experiments"
            ],
            "can_access_public_data": user_permissions[
                "can_access_public_data"
            ],
        }

        return {
            "success": True,
            "user_permissions": user_permissions,
            "metadata_statistics": metadata_stats,
            "accessible_statistics": accessible_stats,
            "timestamp": datetime.now().isoformat(),
        }

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
        # Check access control first
        has_access = False
        if entity_type == "artifact":
            has_access = self.access_control.check_artifact_access(
                entity_id, user_id, "read"
            )
        elif entity_type == "experiment":
            has_access = self.access_control.check_experiment_access(
                entity_id, user_id, "read"
            )

        if not has_access:
            return {
                "success": False,
                "error": f"Access denied: User {user_id} cannot read "
                f"{entity_type} {entity_id}",
                "user_id": user_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "timestamp": datetime.now().isoformat(),
            }

        # Perform compliance validation
        compliance_result = self.access_control.enforce_compliance_policy(
            entity_type, entity_id
        )

        # Get metadata completeness
        completeness_result = (
            self.metadata_manager.validate_metadata_completeness()
        )

        return {
            "success": True,
            "user_id": user_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "access_granted": has_access,
            "compliance_result": compliance_result,
            "completeness_result": completeness_result,
            "timestamp": datetime.now().isoformat(),
        }

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
        # Get access log
        access_log = self.access_control.get_access_log(user_id)

        # Filter by entity type if specified
        if entity_type:
            access_log = [
                entry
                for entry in access_log
                if entry.get("entity_type") == entity_type
            ]

        # Get user permissions
        user_permissions = self.access_control.get_user_permissions(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "entity_type_filter": entity_type,
            "access_log_entries": len(access_log),
            "access_log": access_log,
            "user_permissions": user_permissions,
            "timestamp": datetime.now().isoformat(),
        }

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
        results = {
            "success": True,
            "operation": operation,
            "user_id": user_id,
            "entity_type": entity_type,
            "total_entities": len(entity_ids),
            "processed_entities": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "access_denied": 0,
            "results": [],
            "timestamp": datetime.now().isoformat(),
        }

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
                    results["access_denied"] += 1
                    results["results"].append(
                        {
                            "entity_id": entity_id,
                            "status": "access_denied",
                            "error": f"User {user_id} cannot access "
                            f"{entity_type} {entity_id}",
                        }
                    )
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

                results["processed_entities"] += 1
                if operation_result.get("success", False):
                    results["successful_operations"] += 1
                else:
                    results["failed_operations"] += 1

                results["results"].append(
                    {
                        "entity_id": entity_id,
                        "status": (
                            "success"
                            if operation_result.get("success", False)
                            else "failed"
                        ),
                        "result": operation_result,
                    }
                )

            except Exception as e:
                results["processed_entities"] += 1
                results["failed_operations"] += 1
                results["results"].append(
                    {
                        "entity_id": entity_id,
                        "status": "error",
                        "error": str(e),
                    }
                )

        return results
