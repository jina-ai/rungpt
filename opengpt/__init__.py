import sys

# import importlib.metadata if available, otherwise importlib_metadata
# (for Python < 3.8)
if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    """Return the module version number specified in pyproject.toml.
    :return: The version number.
    """
    return importlib_metadata.version(__package__)


__version__ = get_version()
