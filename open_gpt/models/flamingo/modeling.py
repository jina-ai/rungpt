from typing import List, Optional, Union

import torch
from PIL import Image

from open_gpt.models.modeling import BaseModel


class FlamingoModel(BaseModel):
    image_processor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_and_transforms(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        **kwargs
    ):
        from .loading import load_model_and_transforms

        self.model, self.tokenizer, self.image_processor = load_model_and_transforms(
            model_name_or_path,
            vision_model_name_or_path='ViT-L-14::openai',
            lang_model_name_or_path='facebook/llama-7b',
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self._dtype,
            precision=self._precision,
            device=self._device,
            device_map=self._device_map,
        )

    def generate(self, prompt: str, inplace_images: List = [], **kwargs):
        """Generate text from the given prompt."""

        assert isinstance(prompt, str), "Prompt must be a string."
        prompt_len = len(prompt)

        with torch.inference_mode():
            vision_x = []

            for image in inplace_images:
                vision_x.append(
                    self.image_processor(
                        Image.open(image) if isinstance(image, str) else image
                    ).unsqueeze(0)
                )
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)

            lang_x = self.tokenizer(
                [prompt],
                padding=True,
                return_tensors="pt",
            )

            generated_tokens = self.model.generate(
                vision_inputs=vision_x,
                text_inputs=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20,
                num_beams=6,
                top_p=0.9,
                no_repeat_ngram_size=2,
                temperature=0.7,
                length_penalty=1.5,
            )[0].tolist()

            text = self.tokenizer.decode(
                generated_tokens,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )

            text = text[prompt_len:] if text[:prompt_len] == prompt else text

            return text
