"""Operational helpers extracted from TraceabilityIntegrationManager.

Keep bulk operations and audit/reporting utilities here to slim the manager.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .utils.integration import (
    build_error_response,
    build_success_response,
)


def build_bulk_results_header(
    operation: str, user_id: str, entity_type: str, total: int
) -> dict[str, Any]:
    return {
        "success": True,
        "operation": operation,
        "user_id": user_id,
        "entity_type": entity_type,
        "total_entities": total,
        "processed_entities": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "access_denied": 0,
        "results": [],
        "timestamp": datetime.now().isoformat(),
    }


def append_bulk_result(
    results: dict[str, Any],
    entity_id: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> None:
    results["processed_entities"] += 1
    if status == "success":
        results["successful_operations"] += 1
    elif status == "access_denied":
        results["access_denied"] += 1
    else:
        results["failed_operations"] += 1
    results["results"].append(
        {"entity_id": entity_id, "status": status, "result": payload}
        if payload
        else {"entity_id": entity_id, "status": status}
    )


# --- Query, stats, compliance, and audit helpers --------------------------------


def search_with_access_control(
    access_control: Any,
    metadata_manager: Any,
    metadata_key: str,
    metadata_value: Any,
    user_id: str,
    entity_type: str = "artifact",
) -> dict[str, Any]:
    matching = metadata_manager.search_by_metadata(
        metadata_key, metadata_value, entity_type
    )
    accessible: list[dict[str, Any]] = []
    for entity in matching:
        entity_id = entity.get(f"{entity_type}_id", "")
        if entity_type == "artifact":
            has_access = access_control.check_artifact_access(
                entity_id, user_id, "read"
            )
        elif entity_type == "experiment":
            has_access = access_control.check_experiment_access(
                entity_id, user_id, "read"
            )
        else:
            has_access = True
        if has_access:
            accessible.append(entity)
    return build_success_response(
        {
            "user_id": user_id,
            "metadata_key": metadata_key,
            "metadata_value": metadata_value,
            "entity_type": entity_type,
            "total_matches": len(matching),
            "accessible_matches": len(accessible),
            "accessible_entities": accessible,
        }
    )


def get_metadata_statistics_with_access(
    access_control: Any, metadata_manager: Any, user_id: str
) -> dict[str, Any]:
    user_permissions = access_control.get_user_permissions(user_id)
    metadata_stats = metadata_manager.get_metadata_statistics()
    accessible_stats = {
        "user_id": user_id,
        "total_accessible_artifacts": user_permissions["accessible_artifacts"],
        "total_accessible_experiments": user_permissions[
            "accessible_experiments"
        ],
        "owned_artifacts": user_permissions["owned_artifacts"],
        "owned_experiments": user_permissions["owned_experiments"],
        "metadata_keys_accessible": metadata_stats.get(
            "artifact_metadata_keys", []
        ),
        "can_create_artifacts": user_permissions["can_create_artifacts"],
        "can_create_experiments": user_permissions["can_create_experiments"],
        "can_access_public_data": user_permissions["can_access_public_data"],
    }
    return build_success_response(
        {
            "user_permissions": user_permissions,
            "metadata_statistics": metadata_stats,
            "accessible_statistics": accessible_stats,
        }
    )


def validate_compliance_with_access(
    access_control: Any,
    metadata_manager: Any,
    entity_type: str,
    entity_id: str,
    user_id: str,
) -> dict[str, Any]:
    has_access = False
    if entity_type == "artifact":
        has_access = access_control.check_artifact_access(
            entity_id, user_id, "read"
        )
    elif entity_type == "experiment":
        has_access = access_control.check_experiment_access(
            entity_id, user_id, "read"
        )
    if not has_access:
        return build_error_response(
            f"Access denied: User {user_id} cannot read {entity_type} {entity_id}",
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
        )
    compliance_result = access_control.enforce_compliance_policy(
        entity_type, entity_id
    )
    completeness_result = metadata_manager.validate_metadata_completeness()
    return build_success_response(
        {
            "user_id": user_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "access_granted": has_access,
            "compliance_result": compliance_result,
            "completeness_result": completeness_result,
        }
    )


def audit_trace_with_access_control(
    access_control: Any, user_id: str, entity_type: str | None = None
) -> dict[str, Any]:
    access_log = access_control.get_access_log(user_id)
    if entity_type:
        access_log = [
            entry
            for entry in access_log
            if entry.get("entity_type") == entity_type
        ]
    user_permissions = access_control.get_user_permissions(user_id)
    return build_success_response(
        {
            "user_id": user_id,
            "entity_type_filter": entity_type,
            "access_log_entries": len(access_log),
            "access_log": access_log,
            "user_permissions": user_permissions,
        }
    )
