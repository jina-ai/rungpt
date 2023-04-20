"""The executor wraps the model and provides a simple way to run inference on the model."""

from jina import DocumentArray, Executor, requests

from .models import create_model


class OpenGPTExecutor(Executor):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model = create_model(model_name, **kwargs)

    @requests
    def generate(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.text = self.model.generate(doc.text)
