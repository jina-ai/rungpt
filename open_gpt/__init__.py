import logging as _logging
import os as _os

__version__ = '0.0.10'

_logging.captureWarnings(True)

import jina as _jina

# hide jina user survey
_os.environ['JINA_HIDE_SURVEY'] = '1'


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
