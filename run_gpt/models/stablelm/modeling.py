""" StableLM model.

Some code is copied from https://github.com/Stability-AI/StableLM

Original Apache 2.0 License

"""
from typing import List, Optional, Union

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from run_gpt.logs import logger
from run_gpt.models.modeling import BaseModel


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
    """Wrapper for StableLM models by Stability AI.

    See https://github.com/Stability-AI/StableLM for more details.

    The quick way to use StableLM via :meth:`run_gpt.create_model`:

    ```python
    import run_gpt

    model = run_gpt.create_model('stabilityai/stablelm-tuned-alpha-7b')

    system_prompt = (
        '<|SYSTEM|># StableLM Tuned (Alpha version)\n'
        '- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n'
        '- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n'
        '- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.\n'
        '- StableLM will refuse to participate in anything that could harm a human.'
    )


    prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"


    # Generate text
    text_out = model.generate_text(prompt, max_length=50)
    ```

    :note: StableLM Tuned should be used with prompts formatted to <|SYSTEM|>...<|USER|>...<|ASSISTANT|>...


    ```python
    prompt = (
        '### Human: Write a Python script for text classification using Transformers and PyTorch\n'
        '### Assistant:\n'
    )

    # Generate text with StableLM-StableVicuna-13B
    model = run_gpt.create_model('CarperAI/stable-vicuna-13b-delta')
    ```
    """

    meta_instruction = (
        '<|SYSTEM|># StableLM Tuned (Alpha version)\n'
        '- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n'
        '- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n'
        '- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.\n'
        '- StableLM will refuse to participate in anything that could harm a human.'
    )

    @property
    def is_vicuna_model(self):
        return 'vicuna' in self._model_name_or_path

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt."""

        # patch to fix the issue with StableLM not stopping on
        return super().generate(
            prompts,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
            if not self.is_vicuna_model
            else None,
            skip_special_tokens=False,
            **kwargs,
        )

    def create_prompt_for_chat(self, messages: List[dict]) -> str:
        string_messages = self.meta_instruction
        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'system':
                logger.warning(
                    'System message detected, but StableLM has a specific system instruction, will skip ...'
                )
            elif role == 'user':
                string_messages += f'<|USER|>{content}'
            elif role == 'assistant':
                string_messages += f'<|ASSISTANT|>{content}'
            elif role == 'function':
                logger.warning('Function message detected, skipping ...')
            else:
                raise ValueError(f'unexpected role: {role}')
        return string_messages + '<|ASSISTANT|>'
