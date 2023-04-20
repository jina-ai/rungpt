from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Union

import torch

_PRECISION_TO_DTYPE = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'int8': torch.int8,
}

_DEFAULT_DTYPE = torch.float32


def infer_dtype(precision: Optional[Union[str, 'torch.dtype']]):
    if precision is None:
        return _DEFAULT_DTYPE
    elif isinstance(precision, str):
        return _PRECISION_TO_DTYPE.get(precision, precision)
    elif isinstance(precision, torch.dtype):
        return precision
    else:
        return ValueError(f'Invalid precision: {precision}')


def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()


def utcnow() -> datetime:
    """Return the current utc date and time with tzinfo set to UTC."""
    return datetime.now(timezone.utc)
