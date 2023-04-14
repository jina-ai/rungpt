from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import torch

_PRECISION_TO_DTYPE = {
    'fp16': 'torch.float16',
    'fp32': 'torch.float32',
    'int8': 'torch.int8',
}


def get_dtype(precision: Union[str, 'torch.dtype']):
    return _PRECISION_TO_DTYPE.get(precision, precision)


def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()
