"""The executor wraps the model and provides a simple way to run inference on the model."""

from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Union

import torch
from docarray import DocumentArray
from jina import Executor, requests

from open_gpt.factory import create_model_and_transforms
from open_gpt.logging import logger


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

        logger.info(f'==> loading model of `{model_name_or_path}` ...')
        self.model, self.tokenizer, *_ = create_model_and_transforms(
            model_name_or_path, precision=precision, device_map=device_map, **kwargs
        )

    @requests
    def generate(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
        num_captions = int(parameters.get('num_captions', 1))
        prompted_da = DocumentArray(
            [d for d in docs if d.tags.get('prompt') is not None]
        )
