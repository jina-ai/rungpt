from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

from loguru import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: Union[str, 'torch.dtype'] = 'torch.float16',
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""

    _ = kwargs.pop('device')

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    revision = kwargs.pop('revision', 'step143000')

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        # cache_dir=f"./{model_name_or_path}/step143000",
        device_map="auto",
        **kwargs,
    )

    if str(dtype) == 'torch.float16':
        model.half()
    elif str(dtype) != 'torch.float32':
        raise ValueError(f"Unsupported dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        # cache_dir="./pythia-70m-deduped/step143000",
    )

    return model, tokenizer
