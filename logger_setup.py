import logging
import sys
from typing import Any


def setup_logger(
    name: str, log_file: str = "logs/app.log", to_stdout: bool = False
) -> logging.Logger:
    """Create a logger that always logs to file, and optionally also to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Avoid duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure log directory exists
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler — always on
    file_handler: logging.FileHandler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Optional stdout handler
    if to_stdout:
        stream_handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter: logging.Formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


def enable_stdout(logger: logging.Logger) -> None:
    """Add stdout output dynamically."""
    import sys

    stream_handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def disable_stdout(logger: logging.Logger) -> None:
    """Remove stdout output dynamically."""
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.StreamHandler)
    ]


if __name__ == "__main__":
    import time

    def run_task(logger: logging.Logger) -> None:
        logger.info("Starting task...")
        for i in range(3):
            logger.debug(f"Step {i + 1} running")
            time.sleep(0.5)
        logger.info("Task complete.")

    # Default: logs only to file
    logger: logging.Logger = setup_logger(__name__, log_file="logs/app.log")

    logger.info("Program started (file logging only).")
    run_task(logger)

    # Turn on stdout temporarily
    logger.info("Enabling stdout logging for next task.")
    enable_stdout(logger)

    run_task(logger)

    # Disable stdout again
    logger.info("Disabling stdout logging.")
    disable_stdout(logger)

    logger.info("All tasks finished.")
