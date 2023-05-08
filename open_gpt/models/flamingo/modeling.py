from typing import Optional

from open_gpt.models.modeling import BaseModel


class FlamingoModel(BaseModel):
    no_split_module_classes = ["LlamaDecoderLayer"]
    image_processor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_and_transforms(
        self, model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None
    ):
        from .loading import load_model_and_transforms

        self.model, self.tokenizer, self.image_processor = load_model_and_transforms(
            model_name_or_path,
            vision_model_name_or_path='ViT-L-14::openai',
            lang_model_name_or_path='facebook/llama_7b',
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=self._dtype,
            device=self._device,
            device_map=self._device_map,
            no_split_module_classes=self.no_split_module_classes,
        )

        self.model.eval()
