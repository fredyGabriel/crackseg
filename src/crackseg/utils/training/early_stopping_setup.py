"""Helper for setting up EarlyStopping from config."""

from collections.abc import Mapping
from typing import Any, Protocol

from hydra import errors as hydra_errors  # Import Hydra errors
from hydra.utils import instantiate


# Protocolo mínimo para logger
class LoggerProtocol(Protocol):
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...


def setup_early_stopping(
    cfg: Mapping[str, Any],
    monitor_metric: str,
    monitor_mode: str,
    verbose: bool,
    logger: LoggerProtocol,
) -> object | None:
    """
    Sets up EarlyStopping from config. Returns early_stopper or None.

    This function properly filters configuration parameters to match
    the EarlyStopping constructor signature, avoiding parameter mismatch
    errors.
    """

    def safe_log(
        logger: LoggerProtocol, level: str, *args: Any, **kwargs: Any
    ) -> None:
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(*args, **kwargs)

    early_stopper = None
    early_stopping_cfg = cfg.get("early_stopping", None)
    if early_stopping_cfg:
        # Check if early stopping is explicitly disabled
        if not early_stopping_cfg.get("enabled", True):
            safe_log(
                logger, "info", "Early stopping explicitly disabled in config."
            )
            return None

        try:
            es_monitor = early_stopping_cfg.get("monitor", monitor_metric)
            if not es_monitor.startswith("val_"):
                es_monitor = f"val_{es_monitor}"

            # Filter parameters to match EarlyStopping constructor signature
            # EarlyStopping expects: patience, min_delta, mode, verbose
            allowed_params = {
                "patience",
                "min_delta",
                "mode",
                "verbose",
                "_target_",
            }

            # Create filtered configuration for EarlyStopping instantiation
            filtered_config = {
                k: v
                for k, v in early_stopping_cfg.items()
                if k in allowed_params
            }

            # Override with function parameters if not in config
            if "mode" not in filtered_config:
                filtered_config["mode"] = monitor_mode
            if "verbose" not in filtered_config:
                filtered_config["verbose"] = verbose

            # Instantiate EarlyStopping with filtered parameters
            early_stopper = instantiate(
                filtered_config,
                _recursive_=False,
            )

            # Store the monitor metric information separately for the trainer
            # to use (EarlyStopping doesn't need to know the metric name,
            # just the values)
            early_stopper.monitor_metric = es_monitor
            early_stopper.monitor_mode = filtered_config.get(
                "mode", monitor_mode
            )
            early_stopper.enabled = (
                True  # Mark as enabled since we successfully created it
            )

            safe_log(
                logger,
                "info",
                f"Early stopping enabled. Monitoring: {es_monitor}",
            )
        except (
            hydra_errors.InstantiationException,
            TypeError,
            ValueError,
            AttributeError,
            Exception,
        ) as e:
            safe_log(
                logger,
                "error",
                f"Error initializing EarlyStopping ({type(e).__name__}: {e}).",
                exc_info=True,
            )
            early_stopper = None
    else:
        safe_log(logger, "info", "Early stopping disabled.")
    return early_stopper
