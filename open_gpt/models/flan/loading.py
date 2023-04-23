from typing import TYPE_CHECKING, Optional, Union

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

if TYPE_CHECKING:
    import torch

from loguru import logger


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

    load_in_8bit = True if (str(dtype) == 'torch.int8') else False

    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        torch_dtype=dtype,
        # device_map="auto",
        **kwargs
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path
    )

    return model, tokenizer
