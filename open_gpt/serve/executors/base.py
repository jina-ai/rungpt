"""The executor wraps the model and provides a simple way to run inference on the model."""

from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Union

import torch
from docarray import DocumentArray
from jina import Executor, requests

from open_gpt.factory import create_model
from open_gpt.logs import logger


class CausualLMExecutor(Executor):
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

        self.model = create_model(
            model_name_or_path, precision=precision, device_map=device_map, **kwargs
        )

        # warmup the model to avoid the first-time slowness
        self.model.generate(['Hello world!'])

    @requests(on='/generate')
    def generate(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
        prompted_da = DocumentArray(
            [d for d in docs if d.tags.get('prompt') or d.text is not None]
        )
        prompts = [d.tags['prompt'] or d.text for d in prompted_da]

        if prompts:
            result = self.model.generate(prompts, **parameters)
            for d, r in zip(prompted_da, result):
                d.tags['generated_text'] = r
        else:
            logger.warning('No prompts found in the request.')
