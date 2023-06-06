from typing import List, Optional, Union

import torch

from open_gpt.models.modeling import BaseModel


class LlamaModel(BaseModel):
    no_split_module_classes = ["LlamaDecoderLayer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_and_transforms(
        self, model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None
    ):

        from .loading import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self._dtype,
            precision=self._precision,
            device=self._device,
            device_map=self._device_map,
        )

        self.model.eval()
