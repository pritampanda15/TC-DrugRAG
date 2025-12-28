"""Logging configuration."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = None,
):
    """Setup logging configuration."""
    format_string = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
