from pathlib import Path
from typing import Optional, Union

import torch


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
    precision: Optional[str] = 'fp32',
    **kwargs,
):
    """Create a model of the given name.

    :param model_name: The name of the model to create.
    :param device: The device to create the model on.
    :param precision: The precision to use for the model.
    :param kwargs: Additional arguments to pass to the model.
    :return: The model.
    """
    # TODO: Add support for loading config based on model name
    model_config = {}

    if model_name == 'OpenFlamingo-9B':
        from .models.flamingo.loading import load_model_and_transforms

        model_config = {
            'clip_model_name': 'ViT-L-14::openai',
            'lang_model_name_or_path': 'llama_7B',
            'tokenizer_name_or_path': 'llama_7B',
        }
        return load_model_and_transforms(**model_config)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
