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

    no_split_module_classes = ["GPTNeoXLayer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt."""
        return super().generate(
            prompts, stopping_criteria=StoppingCriteriaList([StopOnTokens()]), **kwargs
        )
