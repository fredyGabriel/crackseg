"""Helper for setting up EarlyStopping from config."""
from hydra.utils import instantiate


def setup_early_stopping(cfg, monitor_metric, monitor_mode, verbose, logger):
    """
    Sets up EarlyStopping from config. Returns early_stopper or None.
    """
    def safe_log(logger, level, *args, **kwargs):
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
                verbose=verbose
            )
            safe_log(
                logger, "info",
                f"Early stopping enabled. Monitoring: {es_monitor}"
            )
        except Exception:
            safe_log(
                logger, "error",
                "Error initializing EarlyStopping.",
                exc_info=True
            )
            early_stopper = None
    else:
        safe_log(logger, "info", "Early stopping disabled.")
    return early_stopper
