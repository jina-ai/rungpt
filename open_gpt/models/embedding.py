from typing import TYPE_CHECKING, List, Union

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class EmbeddingMixin:
    """Mixin for embedding methods."""

    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """Encode a sentence or a list of sentences into a list of embeddings.

        :param sentences: the sentence(s) to encode
        :param normalize_embeddings: whether to normalize embeddings
        :param kwargs: additional arguments to pass to the model
        :return: a list of embeddings in form of numpy arrays
        """
        inputs = self.tokenizer(
            sentences if isinstance(sentences, list) else [sentences],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        model_output = self.model(**inputs)

        embeddings = mean_pooling(model_output, inputs["attention_mask"])

        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.detach().cpu().numpy()

        if isinstance(sentences, str):
            return embeddings[0]

        return embeddings

    def embedding(
        self,
        sentences: Union[str, List[str]],
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """Encode a sentence or a list of sentences into a list of embeddings.

        :param sentences: the sentence(s) to encode
        :param normalize_embeddings: whether to normalize embeddings
        :param kwargs: additional arguments to pass to the model
        :return: a list of embeddings in form of numpy arrays
        """
        return self.encode(sentences, normalize_embeddings, **kwargs)
