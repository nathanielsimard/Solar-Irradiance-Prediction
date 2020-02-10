import logging
import os
import sys

try:
    if os.environ["DEBUG"] == "1":
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
except KeyError:
    LEVEL = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LEVEL)

    return logger
