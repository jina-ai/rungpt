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

    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token
        # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For generation padding tokens should be on the left
    tokenizer.padding_side = "left"

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

    if hasattr(model, 'generation_config'):
        # set pad_token_id to eos_token_id because GPT does not have a PAD token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer
