from typing import List, Optional, Union

import torch

from open_gpt.models.modeling import BaseModel


class LlamaModel(BaseModel):
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
            no_split_module_classes=["LlamaDecoderLayer"],
        )

        self.model.eval()

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
