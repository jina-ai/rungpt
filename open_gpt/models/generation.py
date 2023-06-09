import dataclasses
from typing import TYPE_CHECKING, Iterable, List, Optional, Union, overload

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

MIN_TEMPERATURE = 1e-5
MIN_TOP_P = 1e-8
CONTEXT_LEN = 2048


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


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


class GenerationMixin:
    """Mixin for generation methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    def step_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stream_interval: int = 1,
        **kwargs
    ):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self._device
        )
        input_length = len(input_ids[0])

        logits_processor = prepare_logits_processor(
            temperature,
            repetition_penalty,
            top_p,
            top_k,
        )

        # TODO: pass in the required arguments to the model
        stop_token_ids = []

        stop_token_ids.append(self.tokenizer.eos_token_id)
        stop_token = self.tokenizer.eos_token

        output_ids = input_ids.tolist()[0]

        max_src_len = CONTEXT_LEN - max_new_tokens - 8
        if input_length > max_src_len:
            input_ids = input_ids[:, -max_src_len:]
            input_length = max_src_len

        past_key_values = outputs = logits = next_token = None

        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.model(input_ids, use_cache=True)
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

            if repetition_penalty > 1:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None

            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if temperature < MIN_TEMPERATURE or top_p < MIN_TOP_P:  # greedy
                next_token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(next_token)
            stopped = next_token in stop_token_ids

            if step % stream_interval == 0 or step == max_new_tokens - 1 or stopped:
                tmp_output_ids = output_ids[input_length:]
                rfind_start = 0

                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                partially_stopped = False
                if stop_token:
                    if isinstance(stop_token, str):
                        pos = output.rfind(stop_token, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_token)
                    elif isinstance(stop_token, Iterable):
                        for each_stop in stop_token:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
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
                    pass

            if stopped:
                break

            # finish stream event, which contains finish reason
            if step == max_new_tokens - 1:
                finish_reason = "length"
            elif stopped:
                finish_reason = "stop"
            else:
                finish_reason = None

            yield output, finish_reason, input_length, step

    @overload
    def generate(self, prompt: str, inplace_images: List = [], **kwargs):
        ...

    @overload
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs
    ):
        """Generate text from the given prompt.

        :param prompts: The prompt(s) to generate from.
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
        """
        ...

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        inputs = self.tokenizer(
            [prompts] if isinstance(prompts, str) else prompts,
            padding='longest',
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self._device)

        # overwrite default values with kwargs
        clean_up_tokenization_spaces = kwargs.pop('clean_up_tokenization_spaces', True)
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)

            texts_outs = []
            for _, generated_sequence in enumerate(outputs):
                generated_sequence = generated_sequence.tolist()
                prompt = prompts[_] if isinstance(prompts, list) else prompts
                prompt_len = len(prompt)

                text = self.tokenizer.decode(
                    generated_sequence,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    skip_special_tokens=skip_special_tokens,
                )

                text = text[prompt_len:] if text[:prompt_len] == prompt else text
                texts_outs.append(text.lstrip())

            return texts_outs if isinstance(prompts, list) else texts_outs[0]
