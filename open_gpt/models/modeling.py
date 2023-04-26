from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logging import logger


class OpenGPTModel:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[torch.device] = None,
        device_map: Optional[Union[str, List[int]]] = None,
        **kwargs
    ):
        """Load a model and tokenizer from HuggingFace."""

        from ...helper import auto_dtype_and_device

        dtype, device = auto_dtype_and_device(dtype, device)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        )
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
