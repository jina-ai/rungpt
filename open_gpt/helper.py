from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Union

import torch

_PRECISION_TO_DTYPE = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'int8': torch.int8,
    'torch.float32': torch.float32,
    'torch.float16': torch.float16,
    'torch.int8': torch.int8,
}

_DEFAULT_DTYPE = torch.float32


def cast_torch_dtype(precision: Union[str, 'torch.dtype']):
    assert precision is not None
    if isinstance(precision, str):
        return _PRECISION_TO_DTYPE.get(precision, precision)
    elif isinstance(precision, torch.dtype):
        return precision
    else:
        return ValueError(f'Invalid precision: {precision}')


def cast_precision(dtype: Optional[Union[str, 'torch.dtype']]):
    if isinstance(dtype, str):
        return dtype
    elif dtype == torch.float32:
        return 'fp32'
    elif dtype == torch.float16:
        return 'fp16'
    elif dtype == torch.int8:
        return 'int8'
    else:
        return ValueError(f'Invalid dtype: {dtype} to cast to precision')


def auto_dtype_and_device(
    precision: Union[str, 'torch.dtype'],
    device: Union[str, 'torch.device'],
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if precision is None:
        if device.type == 'cuda':
            dtype = torch.float16
        else:
            dtype = _DEFAULT_DTYPE
    else:
        dtype = cast_torch_dtype(precision)

    return dtype, device


def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()


def utcnow() -> datetime:
    """Return the current utc date and time with tzinfo set to UTC."""
    return datetime.now(timezone.utc)
