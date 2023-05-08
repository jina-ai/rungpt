from typing import List, Optional, Union

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from open_gpt.models.modeling import BaseModel


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StableLMModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_and_transforms(
        self, model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None
    ):
        from ..loading import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self._dtype,
            device=self._device,
            device_map=self._device_map,
            no_split_module_classes=["GPTNeoXLayer"],
        )

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt."""
        return super().generate(
            prompts, stopping_criteria=StoppingCriteriaList([StopOnTokens()]), **kwargs
        )
