import logging
import sys


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
