from typing import TYPE_CHECKING, Optional, Union

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

    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    from ...helper import auto_dtype_and_device

    dtype, device = auto_dtype_and_device(dtype, device)

    revision = kwargs.pop('revision', 'step143000')
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=dtype,
        # cache_dir=f"./{model_name_or_path}/step143000",
        # device_map="auto",
        **kwargs,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        revision=revision,
        # cache_dir="./pythia-70m-deduped/step143000",
    )

    return model, tokenizer
