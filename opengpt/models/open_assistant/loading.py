from typing import TYPE_CHECKING, List, NamedTuple, Union

if TYPE_CHECKING:
    import torch

from loguru import logger

# From https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/utils.py

QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}


class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""


class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}


TOKENIZER_CONFIGS = {
    "galactica": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>")),
    "GPT-JT": TokenizerConfig(
        special_tokens=SpecialTokens(sep_token="<|extratoken_100|>")
    ),
    "codegen": TokenizerConfig(
        special_tokens=SpecialTokens("<|endoftext|>", sep_token="<|endoftext|>")
    ),
    "pythia": TokenizerConfig(
        special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")
    ),
    "gpt-neox": TokenizerConfig(
        special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")
    ),
    "llama": TokenizerConfig(
        special_tokens=SpecialTokens("</s>", "</s>", sep_token="<s>")
    ),
    "cerebras": TokenizerConfig(
        special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", "<|endoftext|>")
    ),
    "deberta-v3": TokenizerConfig(
        special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")
    ),
    "bloom": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>", "<s>")),
    "electra": TokenizerConfig(
        special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")
    ),
}


def match_tokenizer_name(model_name: str) -> TokenizerConfig:
    """
    Match a partial model name to a tokenizer configuration
    i.e. model_name `Salesforce/codegen-2B-multi` has config name `codegen`
    """
    tokenizer_config_matches = [
        config for name, config in TOKENIZER_CONFIGS.items() if name in model_name
    ]
    if not tokenizer_config_matches:
        raise ValueError(
            f"Cannot find any tokeniser configuration to match {model_name=}"
        )
    elif 1 < len(tokenizer_config_matches):
        raise ValueError(
            f"Found multiple tokeniser configuration matches for {model_name=}"
        )
    else:
        return tokenizer_config_matches[0]


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: Union[str, 'torch.dtype'] = 'torch.float16',
    **kwargs,
):
    """Load a model and tokenizer from HuggingFace."""
    import torch
    from tokenizers import pre_tokenizers

    # registers reward model for AutoModel loading
    from . import reward_model
    from .reward_model import GPTNeoXRewardModel

    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    device = kwargs.pop('device')

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import (  # GPTNeoXRewardModel,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GPTNeoXForCausalLM,
        T5ForConditionalGeneration,
    )

    hf_config = AutoConfig.from_pretrained(model_name_or_path)

    model_name = hf_config._name_or_path

    tokenizer_name = model_name_or_path
    if "cerebras" in model_name:
        # Only 13B has a tokenizer available on HF
        tokenizer_name = "cerebras/Cerebras-GPT-13B"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer_config = match_tokenizer_name(model_name)

    if hasattr(hf_config, "per_digit_tokens") and hf_config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if tokenizer_config.special_tokens:
        if "GPT-JT" in model_name:
            tokenizer_config.special_tokens.pad_token = tokenizer.eos_token
        # SpecialTokens : latest in 4.25, 4.26
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer_config.special_tokens.pad_token,
                "eos_token": tokenizer_config.special_tokens.eos_token,
                "sep_token": tokenizer_config.special_tokens.sep_token,
            }
        )

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(
        set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values()))
    )

    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    if "pythia" in model_name:
        model = GPTNeoXRewardModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if hf_config.pooling:
            assert hf_config.pooling in (
                "mean",
                "last",
            ), f"invalid pooling configuration '{hf_config.pooling}'"
            model.config.pooling = hf_config.pooling
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=dtype
        )

    # if str(dtype) == 'torch.float16':
    #     model.half()
    # elif str(dtype) != 'torch.float32':
    #     raise ValueError(f"Unsupported dtype: {dtype}")

    return model, tokenizer
