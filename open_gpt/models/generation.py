import dataclasses
from typing import TYPE_CHECKING, List, Optional, Union, overload

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerationMixin:
    """Mixin for generation methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

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
