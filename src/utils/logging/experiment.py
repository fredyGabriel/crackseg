"""Experiment logger implementation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import psutil
import torch
from omegaconf import DictConfig, OmegaConf

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
            log_dir: Directory to store log files
            experiment_name: Name of the experiment
            config: Optional configuration to log at startup
            log_system_stats: Whether to log system statistics
            log_to_file: Whether to save logs to file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_system_stats = log_system_stats
        self.log_to_file = log_to_file

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = get_logger(
            f"{self.__class__.__name__}.{experiment_name}",
            level=log_level
        )

        if log_to_file:
            self._setup_file_logging()

        # Initialize metrics file
        self.metrics_file = self.log_dir / 'metrics.jsonl'

        # Log initial info
        self.logger.info(
            f"Initialized experiment '{experiment_name}' in {log_dir}"
        )
        if config:
            self.log_config(config)
        if log_system_stats:
            self._log_system_info()

    def _setup_file_logging(self) -> None:
        """Setup file logging handler."""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        )
        self.logger.addHandler(file_handler)

    def _log_system_info(self) -> None:
        """Log system information and hardware stats."""
        try:
            # CPU info
            cpu_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
            }

            # GPU info if available
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'gpu_name': torch.cuda.get_device_name(),
                    'gpu_count': torch.cuda.device_count(),
                    'cuda_version': torch.version.cuda,
                }

                # Get memory for each GPU
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.get_device_properties(i).total_memory
                    gpu_info[f'gpu_{i}_memory'] = mem

            # Log the information
            self.logger.info("System Information:")
            self.logger.info(f"CPU Info: {cpu_info}")
            if gpu_info:
                self.logger.info(f"GPU Info: {gpu_info}")

        except Exception as e:
            self.logger.warning(f"Failed to collect system information: {e}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Metric name/tag
            value: Scalar value
            step: Training step or epoch
        """
        metric_dict = {
            'timestamp': datetime.now().isoformat(),
            'tag': tag,
            'value': float(value),
            'step': step
        }

        # Log to metrics file
        if self.log_to_file:
            with open(self.metrics_file, 'a') as f:
                json.dump(metric_dict, f)
                f.write('\n')

        # Log to console with reduced precision
        self.logger.debug(f"{tag}: {value:.4f} (step {step})")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary or OmegaConf DictConfig
        """
        if isinstance(config, DictConfig):
            # Convert to regular dict for JSON serialization
            config_dict = OmegaConf.to_container(config)
        else:
            config_dict = config

        # Save full config
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Log flattened config for easier viewing
        flat_config = flatten_dict(config_dict)
        self.logger.info("Experiment configuration:")
        for key, value in flat_config.items():
            self.logger.info(f"  {key}: {value}")

    def close(self) -> None:
        """Close the logger and handlers."""
        if self.log_system_stats:
            self._log_system_info()  # Log final system stats

        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
