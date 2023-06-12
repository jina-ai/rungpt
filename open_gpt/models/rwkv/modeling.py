from typing import List, Union

import torch
import torch.nn.functional as F

from open_gpt.models.modeling import BaseModel


class RWKVModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize_embeddings: bool = True,
        **kwargs
    ):

        embeddings = []

        def _layer_hook(module, inp, out):
            embeddings.append(out)

        inputs = self.tokenizer(
            sentences if isinstance(sentences, list) else [sentences],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = inputs.to(self._device)

        with torch.inference_mode():
            hook = self.model.rwkv.embeddings.register_forward_hook(_layer_hook)
            _ = self.model.forward(**inputs)
            hook.remove()

            embeddings = embeddings[0]

            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            embeddings = embeddings.detach().cpu().numpy()

            if isinstance(sentences, str):
                return embeddings[0]

            return embeddings
