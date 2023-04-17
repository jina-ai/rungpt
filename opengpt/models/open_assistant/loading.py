from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

from loguru import logger

# From https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_eval/manual/sampling_report.py
QA_SPECIAL_TOKENS = {
    "Question": "<human>",
    "Answer": "<bot>",
    "StartPrefix": "<prefix>",
    "EndPrefix": "</prefix>",
}
QA_SPECIAL_TOKENS_V2_5 = {
    "prompter": "<|prompter|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "prefix_begin": "<|prefix_begin|>",
    "prefix_end": "<|prefix_end|>",
}


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: Union[str, 'torch.dtype'] = 'torch.float16',
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = kwargs.pop('device')

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        T5ForConditionalGeneration,
    )

    hf_config = AutoConfig.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        **kwargs,
    )

    # if str(dtype) == 'torch.float16':
    #     model.half()
    # elif str(dtype) != 'torch.float32':
    #     raise ValueError(f"Unsupported dtype: {dtype}")

    return model, tokenizer
