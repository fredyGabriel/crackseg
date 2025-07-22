"""
Database seeding and summary functionality for test data provisioning.
This module provides database seeding operations and provisioning
summary reporting functionality.
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import TestDataProvisioner


def seed_test_database(
    self: "TestDataProvisioner", database_path: Path | str
) -> bool:
    """
    Seed a test database with provisioned data. Args: self:
    TestDataProvisioner instance database_path: Path to the database file
    to seed Returns: True if seeding was successful
    """
    try:
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        # Record seeding operation
        seeding_operation = {
            "timestamp": time.time(),
            "database_path": str(database_path),
            "provisioned_items": len(self.provisioned_data),
            "items": list(self.provisioned_data.keys()),
        }

        # Create database content
        db_content: dict[str, Any] = {
            "timestamp": time.time(),
            "provisioned_data": {},
            "seeding_history": self.seeding_history,
        }

        # Convert TestData to JSON-serializable format
        for name, test_data in self.provisioned_data.items():
            db_content["provisioned_data"][name] = {
                "data_type": test_data["data_type"],
                "file_path": str(test_data["file_path"]),
                "metadata": test_data["metadata"],
                "cleanup_required": test_data["cleanup_required"],
            }

        # Write to database file
        with open(database_path, "w") as f:
            json.dump(db_content, f, indent=2)

        # Record this seeding operation
        self.seeding_history.append(seeding_operation)

        self.logger.info(
            f"Seeded test database at {database_path} with "
            f"{len(self.provisioned_data)} items"
        )
        return True

    except Exception as e:
        self.logger.error(f"Failed to seed test database: {e}")
        return False


def get_provisioning_summary(self: "TestDataProvisioner") -> dict[str, Any]:
    """
    Get summary of current provisioning state. Args: self:
    TestDataProvisioner instance Returns: Dictionary with provisioning
    summary
    """
    data_types_count: dict[str, int] = {}
    summary = {
        "total_items": len(self.provisioned_data),
        "data_types": data_types_count,
        "seeding_operations": len(self.seeding_history),
        "recent_operations": (
            self.seeding_history[-5:] if self.seeding_history else []
        ),
    }

    # Count by data type
    for test_data in self.provisioned_data.values():
        data_type = test_data["data_type"]
        data_types_count[data_type] = data_types_count.get(data_type, 0) + 1

    return summary
