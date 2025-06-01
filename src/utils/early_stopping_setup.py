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
        try:
            es_monitor = early_stopping_cfg.get("monitor", monitor_metric)
            if not es_monitor.startswith("val_"):
                es_monitor = f"val_{es_monitor}"
            early_stopper = instantiate(
                early_stopping_cfg,
                _recursive_=False,
                monitor_metric=es_monitor,
                mode=monitor_mode,
                verbose=verbose,
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
