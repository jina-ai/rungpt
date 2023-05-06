from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import nn

from ..helper import auto_dtype_and_device
from .loading import create_model_and_transforms

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel(nn.Module):

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[torch.device] = None,
        device_map: Optional[Union[str, List[int]]] = None,
        **kwargs
    ):
        """Create a model of the given name."""

        super().__init__()

        self._dtype, self._device = auto_dtype_and_device(dtype, device)

        if self._device.type == 'cuda' and device_map is None:
            device_map = 'balanced'
        self._device_map = device_map

        self.load_model_and_transforms(
            model_name_or_path, tokenizer_name_or_path=tokenizer_name_or_path
        )

    def load_model_and_transforms(
        self, model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None
    ):
        self.model, self.tokenizer, *_ = create_model_and_transforms(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self._dtype,
            device=self._device,
            device_map=self._device_map,
        )

        self.model.eval()

    def generate(self, prompt: str, **kwargs):
        """Generate text from the given prompt."""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._device)

        with torch.inference_mode():
            return self.model.generate(**inputs, **kwargs)
