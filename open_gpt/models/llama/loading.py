from typing import List, Optional, Union

import torch

from open_gpt.logs import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    precision: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs,
):
    """Load model and tokenizer from HuggingFace / local.

    :param model_name_or_path: The model id or path to load the model from.
    :param tokenizer_name_or_path: The tokenizer id or path to load the tokenizer from.
    """

    from transformers import AutoModelForCausalLM
    from transformers.models.llama.tokenization_llama import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path or tokenizer_name_or_path
    )

    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token
        # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # For generation padding tokens should be on the left
    tokenizer.padding_side = "left"

    logger.info(f"Loading llama base model from {model_name_or_path}")
    quantization_config = None
    if precision == 'bit8':
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["lm_head", "LlamaDecoderLayer"],
        )
    elif precision == 'bit4':
        from packaging import version
        from transformers import BitsAndBytesConfig

        from open_gpt import importlib_metadata

        trf_version = importlib_metadata.version("transformers")
        if 'dev' in trf_version:
            trf_version = '.'.join(trf_version.split('.')[:-1])
        supports_kbit = version.parse(trf_version) >= version.parse("4.30.0")
        assert supports_kbit, (
            f"Vicuna model k-bit quantization requires transformers >= v4.30.0, you have transformers=={trf_version}.\n"
            f"You can install the latest transformers with `pip install git+https://github.com/huggingface/transformers`."
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype or torch.float16,
        quantization_config=quantization_config,
        device_map={'': device or 0} if (device_map is None) else device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return model, tokenizer
