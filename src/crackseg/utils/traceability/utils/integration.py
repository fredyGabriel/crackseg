"""Helper utilities for traceability integration manager.

Extracted from integration_manager to reduce size and centralize common ops.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_access_log_entry(
    user_id: str, entity_type: str, entity_id: str, action: str, result: str
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "action": action,
        "result": result,
    }


def build_success_response(
    base: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    payload = {"success": True, "timestamp": datetime.now().isoformat()}
    if base:
        payload.update(base)
    payload.update(extra)
    return payload


def build_error_response(message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "success": False,
        "error": message,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra)
    return payload
