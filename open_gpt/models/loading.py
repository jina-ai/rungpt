from typing import List, Optional, Union

import torch
from loguru import logger

from open_gpt.helper import auto_dtype_and_device


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype, device = auto_dtype_and_device(dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path, trust_remote_code=True
    )

    if device_map:
        import huggingface_hub
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        if not os.path.exists(model_name_or_path):
            model_path = huggingface_hub.snapshot_download(model_name_or_path)
        else:
            model_path = model_name_or_path

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=True
            )
            # make sure token embedding weights are still tied if needed
            model.tie_weights()

            model = load_checkpoint_and_dispatch(
                model,
                model_path,
                device_map=device_map,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
        model.to(device)

    return model, tokenizer


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
        from .flamingo.loading import load_model_and_transforms

        model_config = {
            'vision_model_name_or_path': 'ViT-L-14::openai',
            'lang_model_name_or_path': 'llama_7B',
            'tokenizer_name_or_path': 'llama_7B',
        }
        return load_model_and_transforms(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('facebook/llama'):
        from .llama.loading import load_model_and_tokenizer

        model_config = {
            'model_name_or_path': 'llama_7B',
            'tokenizer_name_or_path': 'llama_7B',
        }
        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('google/flan'):
        from .flan.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('EleutherAI/pythia'):
        from .pythia.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('stabilityai/stablelm'):
        from .stablelm.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config
        )
    elif model_name.startswith('fnlp/moss-moon'):
        from .moss.loading import load_model_and_tokenizer

        return load_model_and_tokenizer(
            model_name, device=device, dtype=dtype, **model_config, **kwargs
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')
