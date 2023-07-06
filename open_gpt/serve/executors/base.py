"""The executor wraps the model and provides a simple way to run inference on the model."""

from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Union

from docarray import DocumentArray
from jina import Executor, requests

from open_gpt.factory import create_model
from open_gpt.logs import logger

import pickle


class CausualLMExecutor(Executor):
    def __init__(
        self,
        model_name_or_path: str = '',
        minibatch_size: int = 1,
        adapter_name_or_path: Optional[str] = None,
        device_map: Optional[Union[str, List[int]]] = None,
        precision: Optional[str] = None,
        num_workers: int = 4,
        max_context_length: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            model_name_or_path
        ), '`model_name_or_path` must be provided to initialize the model and tokenizer.'

        self._model_name_or_path = model_name_or_path
        self._adapter_name_or_path = adapter_name_or_path
        self._minibatch_size = minibatch_size
        self._thread_pool = ThreadPool(processes=num_workers)
        self._max_context_length = max_context_length

        self.model = create_model(
            model_name_or_path,
            precision=precision,
            adapter_name_or_path=adapter_name_or_path,
            device_map=device_map,
            **kwargs,
        )

        # warmup the model to avoid the first-time slowness
        self.model.generate('Hello world!', max_new_tokens=64)

    @requests(on='/generate')
    def generate(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
        # TEMPORARY FIX: remove the `__results__` key from the parameters dict
        parameters.pop('__results__', None)
        max_context_length = int(
            parameters.pop('max_context_length', self._max_context_length)
        )

        for k, v in parameters.items():
            if k in ['top_k', 'max_new_tokens', 'num_return_sequences']:
                parameters[k] = int(v)

        for d in docs:
            prompt = d.tags.get('prompt') or d.text
            if not prompt:
                continue

            d.tags['generated_text'] = self.model.generate(
                prompt, max_context_length=max_context_length, **parameters
            )

    @requests(on='/generate_stream')
    def generate_stream(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
        for k, v in parameters.items():
            if k in ['top_k', 'num_return_sequences']:
                parameters[k] = int(v)
            if k == 'max_new_tokens':
                logger.info(f"executor got max_new_tokens: {v} from gateway but it will be ignored")
                parameters.pop('max_new_tokens')

        for d in docs:
            prompt = d.tags.get('input_ids') or d.tags.get('prompt') or d.text
            past_key_values = pickle.loads(d.blob) if len(d.blob) > 0 else None
            if not prompt:
                continue

            generated_text = self.model.step_generate(
                prompt,
                max_new_tokens=1,
                past_key_values=past_key_values,
                **parameters,
            )
            resp = next(generated_text)

            d.blob = pickle.dumps(resp['past_key_values'])
            d.tags['generated_text'] = resp['generated_text']
            d.tags['output_ids'] = resp['output_ids']
            d.tags['usage'] = resp['usage']
            d.tags['finish_reason'] = resp['finish_reason']


