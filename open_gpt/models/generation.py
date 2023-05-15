from typing import TYPE_CHECKING, List, Union

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerationMixin:
    """Mixin for generation methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt."""

        inputs = self.tokenizer(
            [prompts] if isinstance(prompts, str) else prompts,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the correct device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)

            texts_outs = []
            for _, generated_sequence in enumerate(outputs):
                generated_sequence = generated_sequence.tolist()
                prompt = prompts[_] if isinstance(prompts, list) else prompts
                prompt_len = len(prompt)

                text = self.tokenizer.decode(
                    generated_sequence,
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )

                text = text[prompt_len:] if text[:prompt_len] == prompt else text
                texts_outs.append(text)

            return texts_outs
