import os as _os
import sys
import warnings as _warnings

import jina as _jina


def _warning_on_one_line(message, category, filename, lineno, *args, **kwargs):
    return '\033[1;33m%s: %s\033[0m \033[1;30m(raised from %s:%s)\033[0m\n' % (
        category.__name__,
        message,
        filename,
        lineno,
    )


_warnings.formatwarning = _warning_on_one_line
_warnings.simplefilter('ignore', category=DeprecationWarning)

# hide jina user survey
_os.environ['JINA_HIDE_SURVEY'] = '1'


try:
    __jina_version__ = _jina.__version__
except AttributeError as e:
    raise RuntimeError(
        '`jina` dependency is not installed correctly, please reinstall with `pip install -U --force-reinstall jina`'
    )

# import importlib.metadata if available, otherwise importlib_metadata
if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    """Return the module version number specified in pyproject.toml.
    :return: The version number.
    """
    return importlib_metadata.version(__package__ + '_torch')


__version__ = get_version()
