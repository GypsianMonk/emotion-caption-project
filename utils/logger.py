"""Structured logging setup for the project."""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
):
    """
    Configure project-wide logging with console and optional file handler.

    Args:
        log_file:   Optional path to log file
        level:      Logging level (default INFO)
        format_str: Log message format
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "PIL", "matplotlib", "tensorflow"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
