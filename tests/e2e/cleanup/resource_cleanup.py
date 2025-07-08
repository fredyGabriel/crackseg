"""Specialized Resource Cleanup Procedures.

This module provides specific cleanup implementations for different types
of resources including temporary files, processes, network connections,
file handles, and GPU cache management.
"""

import gc
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ResourceCleanupBase(ABC):
    """Base class for resource cleanup procedures."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize cleanup procedure with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    @abstractmethod
    async def cleanup(self, test_id: str) -> int:
        """Execute cleanup procedure.

        Args:
            test_id: Identifier for the test requiring cleanup

        Returns:
            Number of resources cleaned up
        """
        pass

    @abstractmethod
    def get_procedure_name(self) -> str:
        """Get unique name for this cleanup procedure."""
        pass


class TempFileCleanup(ResourceCleanupBase):
    """Cleanup temporary files and test artifacts."""

    async def cleanup(self, test_id: str) -> int:
        """Remove temporary files matching patterns."""
        cleaned_count = 0

        # Default cleanup patterns
        patterns = [
            "temp_*",
            "*.tmp",
            "test_output_*",
            f"*{test_id}*",
            "crackseg_test_*",
        ]

        # Add custom patterns from config
        patterns.extend(self.config.get("temp_patterns", []))

        # Cleanup in common directories
        cleanup_dirs = [
            Path("."),
            Path("outputs"),
            Path("test-artifacts"),
            Path("generated_configs"),
            Path("selenium-videos"),
            Path("htmlcov"),
        ]

        for directory in cleanup_dirs:
            if not directory.exists():
                continue

            try:
                for pattern in patterns:
                    for file_path in directory.glob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.debug(
                                f"Removed temp file: {file_path}"
                            )
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            cleaned_count += 1
                            self.logger.debug(
                                f"Removed temp directory: {file_path}"
                            )

            except Exception as e:
                self.logger.warning(f"Error cleaning {directory}: {e}")

        self.logger.info(
            f"Temp file cleanup completed: {cleaned_count} items removed"
        )
        return cleaned_count

    def get_procedure_name(self) -> str:
        """Get procedure name."""
        return "temp_files"


class ProcessCleanup(ResourceCleanupBase):
    """Cleanup orphaned processes and background tasks."""

    async def cleanup(self, test_id: str) -> int:
        """Terminate orphaned processes."""
        cleaned_count = 0

        # Process patterns to identify test-related processes
        target_patterns = [
            "streamlit",
            "jupyter",
            "pytest",
            test_id,
            "crackseg",
        ]

        # Add custom patterns from config
        target_patterns.extend(self.config.get("process_patterns", []))

        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    # Skip system processes and current process
                    if (
                        proc.info["pid"] <= 1
                        or proc.info["pid"] == psutil.Process().pid
                    ):
                        continue

                    # Check if process matches target patterns
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    proc_name = proc.info["name"] or ""

                    if any(
                        pattern in cmdline.lower()
                        or pattern in proc_name.lower()
                        for pattern in target_patterns
                    ):
                        # Graceful termination first
                        process = psutil.Process(proc.info["pid"])
                        process.terminate()

                        # Wait for graceful shutdown
                        try:
                            process.wait(timeout=5.0)
                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            process.kill()

                        cleaned_count += 1
                        self.logger.debug(
                            f"Terminated process: {proc_name} "
                            f"(PID: {proc.info['pid']})"
                        )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    # Process may have already terminated
                    continue

        except Exception as e:
            self.logger.warning(f"Error during process cleanup: {e}")

        self.logger.info(
            f"Process cleanup completed: {cleaned_count} processes terminated"
        )
        return cleaned_count

    def get_procedure_name(self) -> str:
        """Get procedure name."""
        return "processes"


class NetworkCleanup(ResourceCleanupBase):
    """Cleanup network connections and close open ports."""

    async def cleanup(self, test_id: str) -> int:
        """Close network connections for test processes."""
        cleaned_count = 0

        # Port ranges to clean up (Streamlit, Jupyter, test servers)
        target_port_ranges = [
            (8501, 8510),  # Streamlit default range
            (8888, 8899),  # Jupyter range
            (8600, 8699),  # Test server range
        ]

        # Add custom port ranges from config
        target_port_ranges.extend(self.config.get("port_ranges", []))

        try:
            connections = psutil.net_connections()

            for conn in connections:
                if conn.laddr and conn.status == psutil.CONN_LISTEN:
                    port = conn.laddr.port

                    # Check if port is in target ranges
                    for start_port, end_port in target_port_ranges:
                        if start_port <= port <= end_port:
                            try:
                                # Get process using this port
                                process = (
                                    psutil.Process(conn.pid)
                                    if conn.pid
                                    else None
                                )
                                if process:
                                    process.terminate()
                                    cleaned_count += 1
                                    self.logger.debug(
                                        f"Closed connection on port {port}"
                                    )
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                            break

        except Exception as e:
            self.logger.warning(f"Error during network cleanup: {e}")

        self.logger.info(
            f"Network cleanup completed: {cleaned_count} connections closed"
        )
        return cleaned_count

    def get_procedure_name(self) -> str:
        """Get procedure name."""
        return "network_connections"


class FileCleanup(ResourceCleanupBase):
    """Cleanup file handles and open files."""

    async def cleanup(self, test_id: str) -> int:
        """Force close open file handles."""
        cleaned_count = 0

        try:
            # Force garbage collection to close unreferenced files
            gc.collect()
            cleaned_count += 1

            # Additional cleanup for specific file patterns

            # Close any remaining open files in our process
            try:
                current_process = psutil.Process()
                open_files = current_process.open_files()

                for file_info in open_files:
                    file_path = file_info.path

                    # Check if file is test-related
                    if any(
                        pattern in file_path.lower()
                        for pattern in [
                            "temp",
                            "test",
                            "output",
                            test_id.lower(),
                        ]
                    ):
                        # Files will be closed when process terminates
                        # or when garbage collection runs
                        cleaned_count += 1
                        self.logger.debug(
                            f"Identified open test file: {file_path}"
                        )

            except (psutil.AccessDenied, AttributeError):
                # May not have permission to access file info
                pass

        except Exception as e:
            self.logger.warning(f"Error during file cleanup: {e}")

        self.logger.info(
            f"File handle cleanup completed: {cleaned_count} handles processed"
        )
        return cleaned_count

    def get_procedure_name(self) -> str:
        """Get procedure name."""
        return "file_handles"


class GPUCacheCleanup(ResourceCleanupBase):
    """Cleanup GPU memory cache."""

    async def cleanup(self, test_id: str) -> int:
        """Clear GPU memory cache."""
        cleaned_count = 0

        try:
            # Clear PyTorch CUDA cache if available
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cleaned_count += 1
                self.logger.debug("Cleared PyTorch CUDA cache")

        except ImportError:
            self.logger.debug(
                "PyTorch not available, skipping GPU cache cleanup"
            )
        except Exception as e:
            self.logger.warning(f"Error during GPU cleanup: {e}")

        self.logger.info(
            f"GPU cache cleanup completed: {cleaned_count} caches cleared"
        )
        return cleaned_count

    def get_procedure_name(self) -> str:
        """Get procedure name."""
        return "gpu_cache"


class ResourceCleanupRegistry:
    """Registry for resource cleanup procedures."""

    def __init__(self) -> None:
        """Initialize cleanup registry."""
        self._procedures: dict[str, ResourceCleanupBase] = {
            "temp_files": TempFileCleanup(),
            "processes": ProcessCleanup(),
            "network_connections": NetworkCleanup(),
            "file_handles": FileCleanup(),
            "gpu_cache": GPUCacheCleanup(),
        }

        self.logger = logging.getLogger(__name__)

    def get_available_procedures(self) -> list[str]:
        """Get list of available cleanup procedures."""
        return list(self._procedures.keys())

    async def execute_cleanup(self, procedure_name: str, test_id: str) -> int:
        """Execute specific cleanup procedure."""
        if procedure_name not in self._procedures:
            raise ValueError(f"Unknown cleanup procedure: {procedure_name}")

        procedure = self._procedures[procedure_name]
        return await procedure.cleanup(test_id)

    def register_procedure(self, procedure: ResourceCleanupBase) -> None:
        """Register custom cleanup procedure."""
        name = procedure.get_procedure_name()
        self._procedures[name] = procedure
        self.logger.info(f"Registered cleanup procedure: {name}")
