"""
Visual regression testing module for unified testing framework. This
module provides visual regression testing capabilities, preserving the
unique functionality from the original visual_testing_framework.py.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock


class VisualTestSnapshot:
    """Snapshot of a visual test state."""

    def __init__(
        self,
        test_name: str,
        component_type: str,
        render_output: str,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
        checksum: str = "",
    ) -> None:
        self.test_name = test_name
        self.component_type = component_type
        self.render_output = render_output
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.checksum = checksum or self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum of render output."""
        content = (
            f"{self.test_name}:{self.component_type}:{self.render_output}"
        )
        return hashlib.md5(content.encode()).hexdigest()


class UnifiedVisualTester:
    """Unified visual regression testing from visual_testing_framework.py."""

    def __init__(self, snapshots_dir: Path) -> None:
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._current_snapshots: dict[str, VisualTestSnapshot] = {}

    def capture_component_snapshot(
        self,
        test_name: str,
        component_type: str,
        mock_st: Mock,
        metadata: dict[str, Any] | None = None,
    ) -> VisualTestSnapshot:
        """Capture a snapshot of component render state."""
        render_output = self._extract_render_output(mock_st)
        snapshot = VisualTestSnapshot(
            test_name=test_name,
            component_type=component_type,
            render_output=render_output,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._current_snapshots[test_name] = snapshot
        return snapshot

    def save_snapshot(self, snapshot: VisualTestSnapshot) -> Path:
        """Save snapshot to disk."""
        snapshot_file = self.snapshots_dir / f"{snapshot.test_name}.json"

        snapshot_data = {
            "test_name": snapshot.test_name,
            "component_type": snapshot.component_type,
            "render_output": snapshot.render_output,
            "timestamp": snapshot.timestamp,
            "metadata": snapshot.metadata,
            "checksum": snapshot.checksum,
        }

        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f, indent=2)

        return snapshot_file

    def load_snapshot(self, test_name: str) -> VisualTestSnapshot | None:
        """Load snapshot from disk."""
        snapshot_file = self.snapshots_dir / f"{test_name}.json"

        if not snapshot_file.exists():
            return None

        with open(snapshot_file) as f:
            data = json.load(f)

        return VisualTestSnapshot(
            test_name=data["test_name"],
            component_type=data["component_type"],
            render_output=data["render_output"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )

    def compare_snapshots(
        self, current: VisualTestSnapshot, baseline: VisualTestSnapshot
    ) -> dict[str, Any]:
        """Compare two snapshots and return differences."""
        differences: dict[str, Any] = {
            "checksum_match": current.checksum == baseline.checksum,
            "content_changes": [],
            "metadata_changes": {},
        }

        # Detailed content comparison
        if not differences["checksum_match"]:
            differences["content_changes"] = self._analyze_content_differences(
                current.render_output, baseline.render_output
            )

        return differences

    def _extract_render_output(self, mock_st: Mock) -> str:
        """Extract render output from Streamlit mock calls."""
        output_lines: list[str] = []

        # Extract calls from common Streamlit methods
        methods_to_check = [
            "write",
            "markdown",
            "text",
            "json",
            "code",
            "success",
            "info",
            "warning",
            "error",
        ]

        for method_name in methods_to_check:
            if hasattr(mock_st, method_name):
                method = getattr(mock_st, method_name)
                if method.called:
                    for call in method.call_args_list:
                        args: tuple[Any, ...] = call[0] if call[0] else ()
                        kwargs: dict[str, Any] = call[1] if call[1] else {}
                        output_lines.append(f"{method_name}: {args} {kwargs}")

        return "\n".join(output_lines)

    def _analyze_content_differences(
        self, current_output: str, baseline_output: str
    ) -> list[str]:
        """Analyze content differences between outputs."""
        current_lines = current_output.split("\n")
        baseline_lines = baseline_output.split("\n")

        differences: list[str] = []

        # Simple line-by-line comparison
        max_lines = max(len(current_lines), len(baseline_lines))
        for i in range(max_lines):
            current_line = current_lines[i] if i < len(current_lines) else ""
            baseline_line = (
                baseline_lines[i] if i < len(baseline_lines) else ""
            )

            if current_line != baseline_line:
                differences.append(
                    f"Line {i + 1}: '{current_line}' != '{baseline_line}'"
                )

        return differences
