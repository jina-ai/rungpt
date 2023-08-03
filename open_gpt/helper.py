import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def is_bf16_available():
    try:
        return torch.cuda.is_bf16_supported()
    except:  # noqa
        return False


_DEFAULT_DTYPE = torch.float32
_DEFAULT_FP16_DTYPE = torch.bfloat16 if is_bf16_available() else torch.float16
_DEFAULT_DEVICE_MAP = 'balanced'


_PRECISION_TO_DTYPE = {
    'fp16': _DEFAULT_FP16_DTYPE,
    'fp32': torch.float32,
    'int8': torch.int8,
    'bit8': torch.float16,
    'bit4': torch.float16,
    'float32': torch.float32,
    'float16': torch.float16,
}


def cast_torch_dtype(precision: Union[str, 'torch.dtype']):
    assert precision is not None
    if isinstance(precision, str):
        return _PRECISION_TO_DTYPE.get(precision)
    elif isinstance(precision, torch.dtype):
        return precision
    else:
        return ValueError(f'Invalid precision: {precision}')


def cast_to_precision(dtype: Optional[Union[str, 'torch.dtype']]) -> str:
    if isinstance(dtype, str):
        return dtype
    elif dtype == torch.float32:
        return 'fp32'
    elif dtype == torch.float16:
        return 'fp16'
    elif is_bf16_available() and dtype == torch.bfloat16:
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
            dtype = _DEFAULT_FP16_DTYPE
        else:
            dtype = _DEFAULT_DTYPE
    else:
        dtype = cast_torch_dtype(precision)

    return dtype, device


def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_seeds(seed: int = 32):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def utcnow() -> datetime:
    """Return the current utc date and time with tzinfo set to UTC."""
    return datetime.now(timezone.utc)


def asyncify(f):
    import asyncio
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(f(*args, **kwargs))

    return wrapper


def get_device_map(device):
    if device is None:
        return _DEFAULT_DEVICE_MAP if torch.cuda.is_available() else {'': 'cpu'}

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cpu':
        return {'': 'cpu'}
    elif device.type == 'cuda':
        # bitsandbytes quantization need the device index to be specified
        return {'': f"cuda:{device.index or 0}"}
    else:
        raise ValueError(f"Invalid `device`={device}")
