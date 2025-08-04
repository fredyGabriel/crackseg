"""Callbacks for monitoring GPU-specific metrics like VRAM and utilization."""

from typing import TYPE_CHECKING, Any

from ..exceptions import MonitoringError as MetricCollectionError
from .base import BaseCallback

if TYPE_CHECKING:
    import pynvml  # type: ignore[import-untyped]
else:
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        pynvml = None


class GPUStatsCallback(BaseCallback):
    """
    Callback to collect and log GPU statistics using pynvml.

    Monitors VRAM usage, GPU utilization, and temperature. It safely handles
    the initialization and shutdown of pynvml.
    """

    def __init__(self, device_index: int = 0) -> None:
        """
        Initializes the GPUStatsCallback.

        Args:
            device_index: The index of the GPU device to monitor.
        """
        super().__init__()
        if not self._is_pynvml_available():
            raise ImportError(
                "pynvml is not installed. Please install it to use "
                "GPUStatsCallback."
            )
        self.device_index = device_index
        self.handle = None

    def _is_pynvml_available(self) -> bool:
        """Check if pynvml is available."""
        import importlib.util

        return importlib.util.find_spec("pynvml") is not None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Initializes pynvml and gets the device handle."""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except pynvml.NVMLError as e:
            raise MetricCollectionError(
                f"Failed to initialize NVML: {e}"
            ) from e

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Shuts down pynvml."""
        if self.handle:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                # Log or handle shutdown error, but don't crash
                print(f"Warning: Failed to shut down NVML: {e}")
            self.handle = None

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Collects and logs GPU stats at the end of an epoch."""
        if not self.metrics_manager or not self.handle:
            return

        try:
            # Memory Info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used_gb = float(mem_info.used) / (1024**3)
            mem_total_gb = float(mem_info.total) / (1024**3)
            mem_percent = (
                (float(mem_info.used) / float(mem_info.total)) * 100
                if float(mem_info.total) > 0
                else 0.0
            )

            # Utilization Info
            util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util_percent = float(util_info.gpu)

            # Temperature Info
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )

            metrics_to_log: dict[str, float] = {
                "gpu_vram_used_gb": mem_used_gb,
                "gpu_vram_total_gb": mem_total_gb,
                "gpu_vram_util_percent": mem_percent,
                "gpu_util_percent": gpu_util_percent,
                "gpu_temp_celsius": float(temp),
            }
            self.metrics_manager.log(metrics_to_log)

        except pynvml.NVMLError as e:
            raise MetricCollectionError(
                f"Failed to collect GPU stats: {e}"
            ) from e
