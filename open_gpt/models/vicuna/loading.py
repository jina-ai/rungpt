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
    """Load a model and tokenizer from HuggingFace."""
    import gc
    import json
    import os
    import re

    import huggingface_hub
    import torch

    if precision in ['bit4', 'bit8']:
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

    try:
        model_size = re.search(r".+-(\d+b)-.*", model_name_or_path).groups()[0]
    except:
        raise ValueError(
            f'Cannot parse model size from model name {model_name_or_path}'
        )

    from ..loading import load_model_and_tokenizer as _load_model_and_tokenizer

    model, tokenizer = _load_model_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path or model_name_or_path,
        device=device,
        precision=precision,
        dtype=dtype,
        device_map=device_map,
        **kwargs,
    )

    llama_model_name_or_path = f"decapoda-research/llama-{model_size}-hf"

    logger.info(f"Loading model base weights from {llama_model_name_or_path}")
    if not os.path.exists(llama_model_name_or_path):
        model_path = huggingface_hub.snapshot_download(llama_model_name_or_path)
    else:
        model_path = llama_model_name_or_path

    # Load the index
    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")

    if not os.path.isfile(index_file):
        raise ValueError(f'Cannot find a checkpoint index at {index_file}')

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    for shard_file in shard_files:
        state_dict = torch.load(
            os.path.join(model_path, shard_file), map_location='cpu'
        )

        for name, delta_param in state_dict.items():
            param = model.state_dict()[name]
            delta_param = delta_param.to(param.dtype).to(param.device)

            if (
                name in ['model.embed_tokens.weight', 'lm_head.weight']
                and param.shape[0] > delta_param.shape[0]
            ):  # patch to expend the embedding layer
                tmp_params = torch.zeros_like(
                    param, dtype=param.dtype, device=param.device
                )
                tmp_params[: delta_param.shape[0]] = delta_param
                delta_param = tmp_params

            model.state_dict()[name].data.copy_(param.data + delta_param.data)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # # TODO: This is a hack to quantize the model after loading the weights, and the codes are adapted from
    # # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2680
    # if precision in ['bit4', 'bit8']:
    #     from transformers import BitsAndBytesConfig
    #     from transformers.utils.bitsandbytes import (
    #         get_keys_to_not_convert,
    #         replace_with_bnb_linear,
    #     )
    #
    #     logger.info(f"Quantizing model to {precision} precision")
    #
    #     model.is_quantized = True
    #     keep_in_fp32_modules = getattr(model, '_keep_in_fp32_modules', None) or []
    #
    #     if precision == 'bit8':
    #         model.is_loaded_in_8bit = True
    #         quantization_config = BitsAndBytesConfig(
    #             load_in_8bit=True,
    #             llm_int8_enable_fp32_cpu_offload=True,
    #             llm_int8_skip_modules=["lm_head"],
    #         )
    #     else:
    #         model.is_loaded_in_4bit = True
    #         quantization_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_compute_dtype=torch.bfloat16,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type='nf4',
    #             llm_int8_enable_fp32_cpu_offload=True,
    #             llm_int8_skip_modules=["lm_head", "LlamaDecoderLayer"],
    #         )
    #
    #     llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
    #     load_in_8bit_fp32_cpu_offload = (
    #         quantization_config.llm_int8_enable_fp32_cpu_offload
    #     )
    #
    #     # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    #     if llm_int8_skip_modules is None:
    #         modules_to_not_convert = get_keys_to_not_convert(model) or []
    #     else:
    #         modules_to_not_convert = llm_int8_skip_modules
    #
    #     if not isinstance(modules_to_not_convert, list):
    #         modules_to_not_convert = [modules_to_not_convert]
    #
    #     modules_to_not_convert.extend(keep_in_fp32_modules)
    #
    #     # Extend the modules to not convert to keys that are supposed to be offloaded to `cpu` or `disk`
    #
    #     if isinstance(device_map, dict) and len(device_map.keys()) > 1:
    #         keys_on_cpu = [
    #             key for key, value in device_map.items() if value in ["disk", "cpu"]
    #         ]
    #
    #         if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
    #             raise ValueError(
    #                 "If you want to offload some keys to `cpu` or `disk`, you need to set "
    #                 "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
    #                 " converted to 8-bit but kept in 32-bit."
    #             )
    #
    #         modules_to_not_convert.extend(keys_on_cpu)
    #
    #     model = replace_with_bnb_linear(
    #         model,
    #         modules_to_not_convert=modules_to_not_convert,
    #         quantization_config=quantization_config,
    #     )
    #     model.config.quantization_config = quantization_config

    return model, tokenizer
