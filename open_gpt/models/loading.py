from typing import List, Optional, Union

import torch

from open_gpt.logs import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    use_fast: bool = False,
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(
        f'Loading tokenizer from {tokenizer_name_or_path or model_name_or_path} ...'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=True,
        use_fast=use_fast,
    )

    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # For generation padding tokens should be on the left
    tokenizer.padding_side = "left"

    logger.info(f"Loading {model_name_or_path} with precision {precision} ...")

    quantization_config = None
    if precision == 'bit8':
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["lm_head"],
        )
    elif precision == 'bit4':
        from packaging import version

        from open_gpt import importlib_metadata

        trf_version = importlib_metadata.version("transformers")
        if 'dev' in trf_version:
            trf_version = '.'.join(trf_version.split('.')[:-1])
        supports_kbit = version.parse(trf_version) >= version.parse("4.30.0")
        assert supports_kbit, (
            f"Vicuna model k-bit quantization requires transformers >= v4.30.0, you have transformers=={trf_version}.\n"
            f"You can install the latest transformers with `pip install git+https://github.com/huggingface/transformers`."
        )

        from transformers import BitsAndBytesConfig

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
        # split large weight files into smaller ones and use the disk as temporary storage. This is useful for
        # loading large models on machines with low RAM.
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return model, tokenizer
