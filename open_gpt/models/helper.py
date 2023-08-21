import functools
from typing import List, Union
from transformers import StoppingCriteria
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import torch

MIN_TEMPERATURE = 1e-5
MIN_TOP_P = 1e-8
MAX_LENGTH = 2048


@functools.cache
def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0
    # 1.0 makes it a no-op so we skip two cases.
    if MIN_TEMPERATURE <= temperature < 1:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if MIN_TOP_P <= top_p < 1:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@functools.cache
def get_stop_ids(stop_str: Union[str, List[str]], tokenizer: 'AutoTokenizer'):
    stop_ids = []
    if isinstance(stop_str, str):
        stop_str = [stop_str]
    for stop in stop_str:
        # remove eos token
        ids = tokenizer(stop, add_special_tokens=False)['input_ids']
        stop_ids.append(ids)
    return stop_ids


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_ids:
            if list(input_ids[0][-len(stop_id) :].cpu().numpy()) == stop_id:
                return True
        return False
