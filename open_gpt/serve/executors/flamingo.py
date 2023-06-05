"""The executor wraps the model and provides a simple way to run inference on the model."""

from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Union

import torch
from docarray import DocumentArray
from jina import Executor, requests

import open_gpt
from open_gpt.logs import logger


class FlamingoExecutor(Executor):
    def __init__(
        self,
        model_name_or_path: str = '',
        minibatch_size: int = 1,
        device_map: Optional[Union[str, List[int]]] = None,
        precision: Optional[str] = None,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            model_name_or_path
        ), '`model_name_or_path` must be provided to initialize the model and tokenizer.'

        self._model_name_or_path = model_name_or_path
        self._minibatch_size = minibatch_size
        self._thread_pool = ThreadPool(processes=num_workers)

        # IMPORTANT: flamingo does not allow device_map to be set
        # self._device_map = device_map
        if device_map:
            logger.warning(
                '`device_map` is not supported in FlamingoExecutor. Ignored.'
            )

        self.model = open_gpt.create_model(
            model_name_or_path, precision=precision, device_map=None, **kwargs
        )

        # warmup the model to avoid the first-time slowness
        # self.model.generate(['Hello world!'])

    @requests(on='/generate')
    def generate(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
        """Generate text based on the input text."""
        from .utils import blob2image

        for doc in docs:
            prompt = doc.tags.get('prompt', '') or doc.text
            if not prompt:
                logger.warning('No prompt found in the request.')
                continue

            images = []

            for c in doc.chunks:
                if c.blob:
                    images.append(blob2image(c.blob))
                elif c.uri:
                    c.load_uri_to_blob()
                    images.append(blob2image(c.blob))
                    c.pop('blob')
                else:
                    logger.warning('No image found in the request.')

            result = self.model.generate(prompt, inplace_images=images, **parameters)
            doc.tags['generated_text'] = result
