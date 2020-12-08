"""
Logging utility.
"""

import logging
import sys


def setup_logger():
    """Setup the logger."""
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


LOGGER = setup_logger()
