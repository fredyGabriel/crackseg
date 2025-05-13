"""Experiment logger implementation."""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import psutil
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.experiment_manager import ExperimentManager
from src.utils.logging.base import BaseLogger, get_logger, flatten_dict


class ExperimentLogger(BaseLogger):
    """Logger for tracking experiment metrics, configuration and system stats.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        config: Optional[DictConfig] = None,
        log_system_stats: bool = True,
        log_to_file: bool = True,
        log_level: str = 'INFO'
    ):
        """Initialize the experiment logger.

        Args:
            log_dir: Path to store logs
            experiment_name: Name of the experiment
            config: Configuration for the experiment
            log_system_stats: Whether to log system statistics
            log_to_file: Whether to log to a file
            log_level: Logging level
        """
        super().__init__()

        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config
        self.log_system_stats = log_system_stats

        # Initialize logger - properly using the function parameters
        logger_name = f"experiment.{experiment_name}"
        self.logger = get_logger(name=logger_name, level=log_level)

        # Add file handler if log_to_file is True
        if log_to_file:
            log_file = self.log_dir / f"{experiment_name}.log"
            # Create parent directory if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s][%(name)s] - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")

        # Create experiment manager to handle directory structure
        self.experiment_manager = ExperimentManager(
            base_dir=self.log_dir,
            experiment_name=experiment_name,
            config=config,
            create_dirs=True
        )

        # Get paths from experiment manager
        self.outputs_dir = self.experiment_manager.get_path("experiment")
        self.metrics_dir = self.experiment_manager.get_path("metrics")
        self.config_dir = self.experiment_manager.get_path("config")

        # Define file paths
        self.metrics_file = self.metrics_dir / "metrics.jsonl"

        # Log experiment initialization
        self.logger.info(f"Initialized experiment '{experiment_name}' in \
{log_dir}")
        self.logger.info(f"Metrics will be saved to: {self.metrics_file}")
        self.logger.info(
            f"Configuration will be saved to: "
            f"{self.config_dir / 'config.json'}"
        )

        # Save configuration if provided
        if config is not None:
            self.log_config(config)
            self.logger.info("Experiment configuration:")
            # Print flattened config for readability
            container = OmegaConf.to_container(config, resolve=True)
            if not isinstance(container, dict):
                container = {}
            # Convertir claves a str para cumplir con Dict[str, Any]
            container_str_keys = {str(k): v for k, v in container.items()}
            flat_config = flatten_dict(container_str_keys)
            for key, value in flat_config.items():
                self.logger.info(f"  {key}: {value}")

        # Log system info if requested
        if log_system_stats:
            self.log_system_info()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the metric
            value: Scalar value
            step: Step or epoch number
        """
        metric_dict = {
            "name": tag,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }

        # Log to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metric_dict) + "\n")

        # Log to console
        self.logger.info(f"[{step}] {tag}: {value}")

    def log_metric(self, name: str, value: Any, step: Optional[int] = None,
                   **kwargs) -> None:
        """Log a metric.

        Args:
            name: Metric name
            value: Metric value
            step: Training step or epoch
            **kwargs: Additional metadata
        """
        # If step is provided, use log_scalar to maintain compatibility
        if step is not None:
            self.log_scalar(tag=name, value=float(value), step=step)
            return

        metric_dict = {
            "name": name,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

        # Add any additional metadata
        metric_dict.update(kwargs)

        # Log to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metric_dict) + "\n")

        # Log to console
        self.logger.info(f"Metric '{name}': {value}")

    def log_metrics(self, metrics_dict: Dict[str, Any],
                    step: Optional[int] = None,
                    **kwargs) -> None:
        """Log multiple metrics.

        Args:
            metrics_dict: Dictionary of metrics {name: value}
            step: Training step or epoch
            **kwargs: Additional metadata
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, **kwargs)

    def log_system_info(self) -> None:
        """Log system information."""
        try:
            # CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)

            # Memory info
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)

            # GPU info
            gpu_info = "Not available"
            gpu_memory = "N/A"
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
                gpu_memory = f"{torch.cuda.get_device_properties(
                    0).total_memory / (1024 ** 3):.2f} GB"

            # PyTorch version
            pytorch_version = torch.__version__

            # Log all info
            self.logger.info("System Information:")
            self.logger.info(f"  CPU: {cpu_count} physical cores, \
{cpu_count_logical} logical cores")
            self.logger.info(f"  RAM: {memory_total_gb:.2f} GB total, \
{memory_available_gb:.2f} GB available")
            self.logger.info(f"  CUDA: {gpu_info}")
            self.logger.info(f"  CUDA Memory: {gpu_memory}")
            self.logger.info(f"  PyTorch: {pytorch_version}")

        except Exception as e:
            self.logger.warning(f"Failed to log system info: {str(e)}")

    def log_config(self, config: DictConfig) -> None:
        """Log experiment configuration.

        Args:
            config: Experiment configuration
        """
        # First, save config to file
        self.logger.info("Saving configuration to file")
        config_path = self.experiment_manager.save_config(config)
        self.logger.info(f"Configuration saved to: {config_path}")

    def close(self) -> None:
        """Close the logger and finalize the experiment."""
        self.logger.info(f"Closing logger for experiment \
'{self.experiment_name}'")

        # Update experiment status
        try:
            self.experiment_manager.update_status("completed")
        except Exception as e:
            self.logger.warning(f"Failed to update experiment status: {str(e)}"
                                )

        # Close log handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def log_error(
        self, exception: Exception, context: Optional[str] = None
    ) -> None:
        """Log an error that occurred during the experiment.

        Args:
            exception: The exception that occurred
            context: Additional context about where the error occurred
        """
        # Get traceback
        tb = traceback.format_exc()

        # Build error message
        error_msg = f"Error: {type(exception).__name__}: {str(exception)}"
        if context:
            error_msg = f"Error in {context}: {type(exception).__name__}: \
{str(exception)}"

        # Log to console/file
        self.logger.error(error_msg)
        self.logger.error(f"Traceback:\n{tb}")

        # Update experiment status
        try:
            self.experiment_manager.update_status(
                "failed",
                metadata={
                    "error": {
                        "type": type(exception).__name__,
                        "message": str(exception),
                        "context": context,
                        "traceback": tb
                    }
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to update experiment status: \
{str(e)}")

        # Write error to a specific error log file
        try:
            error_file = self.experiment_manager.get_path("logs") / "error.log"
            with open(error_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {error_msg}\n")
                f.write(f"Traceback:\n{tb}\n\n")
        except Exception as e:
            self.logger.warning(f"Failed to write to error log file: {str(e)}")
