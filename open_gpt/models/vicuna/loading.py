from typing import List, Optional, Union

import torch
from tqdm import tqdm

from open_gpt.logs import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    precision: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    import gc
    import json
    import os

    import huggingface_hub
    import torch

    assert precision not in [
        'bit8',
        'bit4',
    ], 'bit8 and bit4 are not supported for Vicuna models.'

    from ..llama.loading import (
        load_model_and_tokenizer as llama_load_model_and_tokenizer,
    )

    model_size = model_name_or_path.split('-')[-3]

    llama_model_name_or_path = f"decapoda-research/llama-{model_size}-hf"

    logger.info(
        f"Loading llama-{model_size} base model from {llama_model_name_or_path}"
    )

    model, tokenizer = llama_load_model_and_tokenizer(
        llama_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        device=device,
        precision='fp16',
        dtype=dtype,
        device_map=device_map,
        **kwargs,
    )

    logger.info(f"Loading model weights delta from {model_name_or_path}")
    if not os.path.exists(model_name_or_path):
        model_path = huggingface_hub.snapshot_download(model_name_or_path)
    else:
        model_path = model_name_or_path

    # Load the index
    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")

    if not os.path.isfile(index_file):
        raise ValueError(f'Cannot find a checkpoint index at {index_file}')

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    for shard_file in shard_files:
        state_dict = torch.load(
            os.path.join(model_path, shard_file), map_location='cpu'
        )

        for name, delta_param in state_dict.items():
            param = model.state_dict()[name]
            delta_param = delta_param.to(param.dtype).to(param.device)

            model.state_dict()[name].data.copy_(param.data + delta_param.data)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    return model, tokenizer
