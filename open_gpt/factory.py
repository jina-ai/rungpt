from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger

from .helper import auto_dtype_and_device


def list_models():
    """List the available models."""
    ...


def load_state_dict(model: torch.nn.Module, checkpoint: Union[str, 'Path']):
    """Load a model state dict.

    :param model: The model to load the state dict into.
    :param checkpoint: The checkpoint to load the state dict from. It can be a path to a checkpoint file/folder or a URL.
    """
    ...


def create_model_and_transforms(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    precision: Optional[str] = None,
    **kwargs,
):
    """Create a model of the given name.

    :param model_name: The name of the model to create.
    :param device: The device to create the model on.
    :param precision: The precision to use for the model.
    :param kwargs: Additional arguments to pass to the model.
    :return: The model.
    """

    dtype, device = auto_dtype_and_device(precision, device)

    # TODO: Add support for loading config based on model name
    model_config = {}

    logger.info(
        f'Loading "{model_name}" with precision: `{dtype}` on device: `{device}`'
    )

    if model_name.startswith('openflamingo/OpenFlamingo'):
        from .models.flamingo.loading import load_model_and_transforms

        model_config = {
            'vision_model_name_or_path': 'ViT-L-14::openai',
            'lang_model_name_or_path': 'llama_7B',
            'tokenizer_name_or_path': 'llama_7B',
        }
        return load_model_and_transforms(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('facebook/llama'):
        from .models.llama.loading import load_model_and_tokenizer

        model_config = {
            'model_name_or_path': 'llama_7B',
            'tokenizer_name_or_path': 'llama_7B',
        }
        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('google/flan'):
        from .models.flan.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('EleutherAI/pythia'):
        from .models.pythia.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')
