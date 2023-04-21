from typing import Optional, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    **kwargs
):
    """Load a model and tokenizer from HuggingFace."""

    from ...helper import auto_dtype_and_device

    dtype, device = auto_dtype_and_device(dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path, local_files_only=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=False,
    )
    model.to(device)

    return model, tokenizer
