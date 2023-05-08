from typing import List, Optional, Union

import torch

from open_gpt.models.modeling import BaseModel


class LlamaModel(BaseModel):
    no_split_module_classes = ["LlamaDecoderLayer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt."""

        prompts = [prompts] if isinstance(prompts, str) else prompts

        with torch.inference_mode():
            texts_outs = []

            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                )
                # Move inputs to the correct device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._device)

                inputs.pop('token_type_ids')

                outputs = self.model.generate(**inputs, **kwargs)[0].tolist()

                prompt_len = len(prompt)

                text = self.tokenizer.decode(
                    outputs,
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )

                text = text[prompt_len:] if text[:prompt_len] == prompt else text
                texts_outs.append(text)

            return texts_outs
