import sys

from loguru import logger
from loguru._defaults import LOGURU_FORMAT


def setup_logging(debug: bool):
    """
    Setup the log formatter for AnnLite.
    """

    log_level = 'INFO'
    if debug:
        log_level = 'DEBUG'

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level=log_level,
    )
