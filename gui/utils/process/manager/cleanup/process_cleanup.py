"""Process cleanup and termination utilities.

This module provides utilities for graceful and force termination of
training processes, including process tree cleanup and zombie process
handling.
"""

import os
import signal
import time
from typing import Any

import psutil


class ProcessCleanup:
    """Handle process cleanup and termination."""

    def terminate_gracefully(self, process: Any, timeout: float) -> bool:
        """Terminate process gracefully.

        Args:
            process: The subprocess to terminate
            timeout: Maximum time to wait for termination

        Returns:
            True if process terminated successfully, False otherwise
        """
        if process is None or process.poll() is not None:
            return True

        try:
            # Send SIGTERM
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()

            # Wait for termination
            start_time = time.time()
            while time.time() - start_time < timeout:
                if process.poll() is not None:
                    return True
                time.sleep(0.1)

            return False

        except Exception:
            return False

    def force_kill(self, process: Any) -> None:
        """Force kill the process.

        Args:
            process: The subprocess to kill
        """
        if process is None or process.poll() is not None:
            return

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass

    def kill_process_tree_nuclear(self) -> int:
        """Nuclear cleanup of process tree.

        Returns:
            Number of processes killed
        """
        killed_count = 0

        # Clean up PyTorch processes
        killed_count += self._cleanup_pytorch_processes()

        # Clean up zombies
        killed_count += self._cleanup_zombies()

        return killed_count

    def _cleanup_pytorch_processes(self) -> int:
        """Clean up PyTorch-related processes.

        Returns:
            Number of processes killed
        """
        killed_count = 0

        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    # Check if it's a Python process
                    if (
                        proc.info["name"]
                        and "python" in proc.info["name"].lower()
                    ):
                        cmdline = proc.info["cmdline"]
                        if cmdline and any(
                            "main.py" in arg for arg in cmdline
                        ):
                            if self._is_related_process(proc):
                                proc.terminate()
                                killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

        return killed_count

    def _cleanup_zombies(self) -> int:
        """Clean up zombie processes.

        Returns:
            Number of zombies cleaned
        """
        cleaned_count = 0

        try:
            for proc in psutil.process_iter(["pid", "status"]):
                try:
                    if proc.info["status"] == psutil.STATUS_ZOMBIE:
                        # Try to reap the zombie
                        try:
                            proc.wait(timeout=0.1)
                            cleaned_count += 1
                        except psutil.TimeoutExpired:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

        return cleaned_count

    def _is_related_process(self, proc: psutil.Process) -> bool:
        """Check if process is related to our training.

        Args:
            proc: Process to check

        Returns:
            True if process is related, False otherwise
        """
        try:
            cmdline = proc.cmdline()
            if not cmdline:
                return False

            # Check for training-related keywords
            training_keywords = [
                "main.py",
                "train",
                "training",
                "crackseg",
                "hydra",
                "python",
                "torch",
            ]

            cmdline_str = " ".join(cmdline).lower()
            return any(keyword in cmdline_str for keyword in training_keywords)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def get_process_tree_info(self, process: Any) -> dict[str, Any]:
        """Get detailed information about the process tree.

        Args:
            process: The subprocess to analyze

        Returns:
            Dictionary with process tree information
        """
        if process is None:
            return {"error": "No active process"}

        try:
            proc = psutil.Process(process.pid)
            children = proc.children(recursive=True)

            tree_info = {
                "main_process": {
                    "pid": proc.pid,
                    "name": proc.name(),
                    "cmdline": proc.cmdline(),
                    "memory_info": proc.memory_info()._asdict(),
                    "cpu_percent": proc.cpu_percent(),
                },
                "children": [],
                "total_processes": len(children) + 1,
            }

            for child in children:
                try:
                    child_info = {
                        "pid": child.pid,
                        "name": child.name(),
                        "cmdline": child.cmdline(),
                        "memory_info": child.memory_info()._asdict(),
                        "cpu_percent": child.cpu_percent(),
                    }
                    tree_info["children"].append(child_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return tree_info

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return {"error": f"Process access error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
