from typing import TYPE_CHECKING, Union

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    import torch

from loguru import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    dtype: Union[str, 'torch.dtype'] = 'torch.float16',
    **kwargs
):
    """Load a model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, local_files_only=True
    )

    # Create a model and initialize it with empty weights
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Load the checkpoint and dispatch it to the right devices
    model = load_checkpoint_and_dispatch(
        model, model_name_or_path, device_map="auto", dtype=dtype, **kwargs
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path, local_files_only=False
    # )
    return model, tokenizer
