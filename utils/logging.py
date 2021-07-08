import logging
import sys


def setup_logger(stream=sys.stderr):
    """Setup a logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


LOGGER = setup_logger()
