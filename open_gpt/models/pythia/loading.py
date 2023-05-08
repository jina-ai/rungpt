from typing import List, Optional, Union

import torch
from loguru import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs
):
    """Load a model and tokenizer from HuggingFace."""
    import os

    import huggingface_hub
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GPTNeoXForCausalLM,
    )

    from ...helper import auto_dtype_and_device

    dtype, device = auto_dtype_and_device(dtype, device)
    revision = kwargs.pop('revision', 'step143000')

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        revision=revision,
    )

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if device_map:
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
                dtype=dtype,
                no_split_module_classes=["GPTNeoXLayer"],
            )
    else:

        model = GPTNeoXForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=dtype,
            **kwargs,
        )
        model.to(device)

    return model, tokenizer
