import functools
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union, overload

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

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


class GenerationMixin:
    """Mixin for generation methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    @torch.inference_mode()
    def step_generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[List[int]] = None,
        max_new_tokens: Optional[int] = None,
        completion_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        max_length: int = MAX_LENGTH,
        echo: bool = False,
        stream_interval: int = 1,
        stop_str: Optional[str] = None,
        stop_token_ids: List[int] = [],
        past_key_values: Optional[Iterable[torch.Tensor]] = None,
        **kwargs,
    ):
        """Generate tokens in a streaming fashion. This method is a modified version of `fastchat.server.inference.generate_stream`.

        :param prompt: The prompt is the context that the model will use to generate the response.
        :param input_ids: The input ids to use for generation.
        :param max_new_tokens: The maximum number of tokens to generate. If None, the model will generate until it predicts a stop token.
        :param completion_tokens: The number of tokens that has been generated before.
        :param temperature: The temperature to use when sampling from the logits.
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.
        :param repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        :param max_length: Maximum length of the sequence which includes prompt and generated text. If the context is longer than this, it will be truncated.
        :param echo: If True, the prompt will be included in the generated response.
        :param stream_interval: The number of tokens to generate before returning the generated tokens.
        :param stop_str: If not None, the model will stop generating when the generated tokens end with this string.
        :param stop_token_ids: A list of token ids that will cause the model to stop generating.
        :param past_key_values: A list of past key values to use for generation. If None, the model will generate from scratch.
        :param kwargs: Additional keyword arguments to pass to the model.
        :return: A dictionary contains generated text, output ids, past_key_values, finish reason and usage information.
                usage information: {'completion_tokens': int, how many tokens have been generated.
                                    'input_length': int, the length of input ids or prompt.
                                    'prompt_tokens': int, the length of past_key_values passed to model.forward() when generating this token.
                                    'total_tokens': int, completed_tokens + input_tokens.
                                }
        """

        def _get_finish_reason(step, completion_tokens, max_new_tokens):
            if step + 1 + completion_tokens == max_new_tokens:
                return "length"
            elif stopped:
                return "stop"
            else:
                return None

        if prompt is None and input_ids is None:
            raise ValueError("Either prompt or input_ids must be provided.")
        if prompt and input_ids:
            raise ValueError("Only one of prompt or input_ids can be provided.")

        if input_ids is None:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                self._device
            )
            len_prompt = len(input_ids[0])
        else:
            assert isinstance(input_ids[0], int), (
                f"input_ids must be list of int, " f"got list of {type(input_ids[0])}"
            )
            len_prompt = None
            input_ids = torch.Tensor([input_ids]).to(dtype=int).to(self._device)

        input_length = len(input_ids[0])

        logits_processor = prepare_logits_processor(
            temperature,
            repetition_penalty,
            top_p,
            top_k,
        )

        stop_token_ids.append(self.tokenizer.eos_token_id)

        output_ids = input_ids.tolist()[0]

        # TODO: figure out what 8 means here
        max_src_len = max_length - max_new_tokens - 8
        if input_length > max_src_len:
            input_ids = input_ids[:, -max_src_len:]
            input_length = max_src_len

        next_token = None

        for step in range(max_new_tokens):
            prompt_tokens = past_key_values[0][0].shape[2] if past_key_values else 0
            if step == 0:
                outputs = self.model(
                    input_ids, use_cache=True, past_key_values=past_key_values
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                outputs = self.model(
                    input_ids=torch.as_tensor([[next_token]], device=self._device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[
                    0
                ]
            else:
                last_token_logits = logits[0, -1, :]

            if last_token_logits.device.type == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < MIN_TEMPERATURE or top_p < MIN_TOP_P:  # greedy
                next_token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(next_token)
            stopped = next_token in stop_token_ids

            if (
                step % stream_interval == 0
                or step + completion_tokens == max_new_tokens - 1
                or stopped
            ):
                tmp_output_ids = output_ids[input_length:]

                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, 0)
                        if pos != -1:
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, 0)
                            if pos != -1:
                                stopped = True
                                break
                            else:
                                partially_stopped = partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # prevent yielding partial stop sequence
                if not partially_stopped:
                    yield {
                        "choices": [
                            {
                                "index": step,
                                "text": output,
                                "finish_reason": _get_finish_reason(
                                    step, completion_tokens, max_new_tokens
                                ),
                            },
                        ],
                        "prompt": prompt,
                        "output_ids": tmp_output_ids,
                        "past_key_values": past_key_values,
                        "usage": {
                            "prompt_tokens": prompt_tokens + input_length,
                            "completion_tokens": completion_tokens + step + 1,
                            "total_tokens": (prompt_tokens + input_length)
                            + (completion_tokens + step + 1),
                        },
                    }

            if stopped:
                break

        yield {
            "choices": [
                {
                    "index": step,
                    "text": output,
                    "finish_reason": _get_finish_reason(
                        step, completion_tokens, max_new_tokens
                    ),
                },
            ],
            "prompt": prompt,
            "output_ids": tmp_output_ids,
            "past_key_values": past_key_values,
            "usage": {
                "prompt_tokens": prompt_tokens + input_length,
                "completion_tokens": completion_tokens + step + 1,
                "total_tokens": (prompt_tokens + input_length)
                + (completion_tokens + step + 1),
            },
        }

    @overload
    def generate(self, prompt: str, inplace_images: List = [], **kwargs):
        ...

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        echo: bool = False,
        **kwargs,
    ):
        """Generate text from the given prompt.

        :param prompt: The prompt input text.
        :param max_new_tokens: The maximum number of tokens to generate, not including the prompt.
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param do_sample: Whether to use sampling instead of greedy decoding.
        :param temperature: The temperature to use for sampling. Only relevant if do_sample is True. Higher means more stochastic.
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. Only relevant if do_sample is True.
        :param top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Only relevant if do_sample is True.
        :param repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty.
        :param length_penalty: Exponential penalty to the length that is used with beam-based generation.
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
                Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences,
                while length_penalty < 0.0 encourages shorter sequences.
        :param no_repeat_ngram_size: If set to int > 0, all ngrams of that size can only occur once.
        :param echo: Whether to echo the prompt in the generated text.
        """
        ...

    def generate(self, prompt: str, max_length: int = MAX_LENGTH, **kwargs):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self._device)

        input_length = inputs["input_ids"].shape[-1]

        if 'stop_str' in kwargs:
            stop_str = kwargs.pop('stop_str')
            stop_ids = get_stop_ids(stop_str, self.tokenizer)
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            stopping_criteria = None

        # overwrite default values with kwargs
        clean_up_tokenization_spaces = kwargs.pop('clean_up_tokenization_spaces', True)
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        echo = kwargs.pop("echo", False)

        max_length = kwargs.pop("max_length", max_length)
        max_new_tokens = kwargs.pop("max_new_tokens", max_length - input_length - 1)

        if max_new_tokens + input_length >= max_length:
            raise ValueError(
                f"max_new_tokens + input_length ({max_new_tokens + input_length}) must be less than max_length ({max_length})"
            )

        kwargs.update(
            {
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "stopping_criteria": stopping_criteria,
            }
        )

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)[0].tolist()
            text = self.tokenizer.decode(
                outputs if echo else outputs[input_length:],
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                skip_special_tokens=skip_special_tokens,
            )

            finish_reason = (
                "length" if len(outputs) - input_length == max_new_tokens else "stop"
            )
            resp = {
                "choices": [
                    {"index": 0, "text": text, "finish_reason": finish_reason},
                ],
                "prompt": prompt,
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": len(outputs),
                    "total_tokens": input_length + len(outputs),
                },
            }

            return resp
