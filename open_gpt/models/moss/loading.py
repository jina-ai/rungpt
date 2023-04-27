import os
from typing import List, Optional, Union

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs
):
    """Load a model and tokenizer from HuggingFace."""

    from ...helper import auto_dtype_and_device

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
                no_split_module_classes=["MossBlock"],
                dtype=dtype,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
        model.to(device)

    return model, tokenizer
