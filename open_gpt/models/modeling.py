from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..factory import create_model_and_transforms
from ..helper import auto_dtype_and_device
from ..logging import logger


def create_model(model_name_or_path: str, **kwargs) -> 'OpenGPTModel':
    """Create a model of the given name.

    :param model_name_or_path: The name or path of the model to create.
    :param kwargs: Additional arguments to pass to the model.
    :return: The model.
    """
    return OpenGPTModel(model_name_or_path, **kwargs)


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

        self.dtype, self.device = auto_dtype_and_device(dtype, device)

        if self.device.type == 'cuda' and device_map is None:
            device_map = 'balanced'

        self.model, self.tokenizer, *_ = create_model_and_transforms(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self.dtype,
            device=self.device,
            device_map=device_map,
        )

        self.model.eval()

    def generate(self, **kwargs):
        """Generate a sequence from the model."""

        with torch.inference_mode():
            return self.model.generate(**kwargs)
