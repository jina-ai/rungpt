from typing import List, Optional, Union

import torch
from tqdm import tqdm

from open_gpt.helper import auto_dtype_and_device
from open_gpt.logging import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    precision: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_size = model_name_or_path.split('-')[-3]

    facebook_llama_model_name_or_path = f"facebook/llama-{model_size}"
    if os.path.exists(facebook_llama_model_name_or_path):
        llama_model_name_or_path = facebook_llama_model_name_or_path
    else:
        llama_model_name_or_path = f"decapoda-research/llama-{model_size}-hf"

    logger.info(
        f"Loading llama-{model_size} base model from {llama_model_name_or_path}"
    )

    from ..llama.loading import (
        load_model_and_tokenizer as llama_load_model_and_tokenizer,
    )

    model, tokenizer = llama_load_model_and_tokenizer(
        llama_model_name_or_path,
        device=device,
        precision=precision,
        dtype=dtype,
        device_map=device_map,
        no_split_module_classes=no_split_module_classes,
        **kwargs,
    )

    logger.info(f"Loading model weights delta from {model_name_or_path}")
    delta = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    # adapted from `fastchat.model.apply_delta`
    for name, param in tqdm(model.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name].to(param.dtype).to(param.device)

    # clean up delta to save memory
    delta = None
    torch.cuda.empty_cache()

    return model, tokenizer
