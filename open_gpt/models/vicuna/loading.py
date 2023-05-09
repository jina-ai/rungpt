from typing import List, Optional, Union

import torch
from tqdm import tqdm

from open_gpt.helper import auto_dtype_and_device
from open_gpt.logging import logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype, device = auto_dtype_and_device(dtype, device)

    model_size = model_name_or_path.split('-')[-3]

    facebook_llama_model_name_or_path = f"facebook/llama-{model_size}"
    if os.path.exists(facebook_llama_model_name_or_path):
        llama_model_name_or_path = facebook_llama_model_name_or_path
    else:
        llama_model_name_or_path = f"decapoda-research/llama-{model_size}-hf"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path or tokenizer_name_or_path, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token
        # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For generation padding tokens should be on the left
    tokenizer.padding_side = "left"

    logger.info(
        f"Loading llama-{model_size} base model from {llama_model_name_or_path}"
    )
    if device_map:
        import huggingface_hub
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        if not os.path.exists(llama_model_name_or_path):
            model_path = huggingface_hub.snapshot_download(llama_model_name_or_path)
        else:
            model_path = llama_model_name_or_path

        with init_empty_weights():
            config = AutoConfig.from_pretrained(
                llama_model_name_or_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=True
            )
            # make sure token embedding weights are still tied if needed
            model.tie_weights()

            model = load_checkpoint_and_dispatch(
                model,
                model_path,
                device_map=device_map,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            llama_model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
        model.to(device)

    logger.info(f"Loading model weights delta from {model_name_or_path}")
    delta = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    # adapted from `fastchat.model.apply_delta`
    for name, param in tqdm(model.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name].to(param.dtype).to(param.device)

    # clean up delta to save memory
    delta = None
    torch.cuda.empty_cache()

    return model, tokenizer
