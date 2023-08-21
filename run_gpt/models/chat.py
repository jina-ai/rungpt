from typing import List, Optional

import torch

MAX_LENGTH = 2048


class ChatMixin:
    """Mixin for chat methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    @torch.inference_mode()
    def chat(
        self,
        messages: List[dict],
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
        **kwargs
    ):
        """Generate text from the given prompt.

        :param messages: A list of messages comprising the conversation so far.
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

        # normalize input
        prompt = self.create_prompt_for_chat(messages)
        completion_response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            echo=echo,
            **kwargs
        )
        # normalize output
        choices = completion_response.pop('choices')
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': choices[0]['text']},
                    'finish_reason': choices[0]['finish_reason'],
                }
            ],
            **completion_response,
        }

    @torch.inference_mode()
    def step_chat(
        self,
        messages: Optional[List[dict]] = None,
        input_ids: Optional[List[int]] = None,
        **kwargs
    ):
        if messages is None and input_ids is None:
            raise ValueError("Either messages or input_ids must be provided.")
        if messages and input_ids:
            raise ValueError("Only one of messages or input_ids can be provided.")

        if messages:
            # normalize input
            prompt = self.create_prompt_for_chat(messages)
            completion_response = self.step_generate(prompt=prompt, **kwargs)
        else:
            completion_response = self.step_generate(input_ids=input_ids, **kwargs)
        # normalize output
        for response in completion_response:
            choices = response.pop('choices')
            yield {
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': choices[0]['text']},
                        'finish_reason': choices[0]['finish_reason'],
                    }
                ],
                **response,
            }
