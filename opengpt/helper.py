import sys

from loguru import logger


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


def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()
