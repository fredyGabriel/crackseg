#!/usr/bin/env python
"""
Evaluation script for trained crack segmentation models. This is a
wrapper script that redirects to the main evaluation module. For
direct usage, consider using 'python -m src.evaluation' instead.
Usage: python -m src.evaluate --checkpoint /path/to/checkpoint.pth.tar
--config /path/to/config.yaml
"""

from crackseg.evaluation.__main__ import main
from crackseg.utils.logging import get_logger

# Configure logger
log = get_logger("evaluate_wrapper")

if __name__ == "__main__":
    log.info(
        "Running evaluation via wrapper "
        "(consider using 'python -m src.evaluation' directly)"
    )
    main()
