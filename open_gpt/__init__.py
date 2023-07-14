import logging as _logging
import os as _os
import sys as _sys

__version__ = '0.0.12'

_logging.captureWarnings(True)

import jina as _jina

# hide jina user survey
_os.environ['JINA_HIDE_SURVEY'] = '1'


# import importlib.metadata if available, otherwise importlib_metadata
if _sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata  # noqa: F401
else:
    import importlib_metadata  # noqa: F401

try:
    __jina_version__ = _jina.__version__
except AttributeError as e:
    raise RuntimeError(
        '`jina` dependency is not installed correctly, please reinstall with `pip install -U --force-reinstall jina`'
    )

__resources_path__ = _os.path.join(_os.path.dirname(__file__), 'resources')

_os.environ['NO_VERSION_CHECK'] = '1'

from inference_client import Client  # noqa

from .factory import create_model
