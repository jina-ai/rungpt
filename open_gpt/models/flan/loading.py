from typing import TYPE_CHECKING, Union

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

if TYPE_CHECKING:
    import torch

from loguru import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: Union[str, 'torch.dtype'] = 'torch.float16',
    **kwargs
):
    """Load a model and tokenizer from HuggingFace."""

    _ = kwargs.pop('device')

    load_in_8bit = True if (str(dtype) == 'torch.int8') else False

    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        torch_dtype=dtype if not load_in_8bit else None,
        device_map="auto",
        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer
