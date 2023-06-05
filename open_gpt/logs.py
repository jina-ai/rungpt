import os
import sys

from loguru import logger


def setup_logger():
    logging_level = os.environ.get("LOG_LEVEL", "INFO")

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
        level=logging_level,
    )
    logger.level("WARNING", color="<fg #d3d3d3>")
    return logger


logger = setup_logger()
