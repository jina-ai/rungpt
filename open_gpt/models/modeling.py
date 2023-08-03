from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import nn

from ..helper import auto_dtype_and_device, get_device_map
from .chat import ChatMixin
from .embedding import EmbeddingMixin
from .generation import GenerationMixin

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel(nn.Module, GenerationMixin, ChatMixin, EmbeddingMixin):
    model: 'AutoModelForCausalLM'
    tokenizer: 'AutoTokenizer'

    def __init__(
        self,
        model_name_or_path: str,
        adapter_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        precision: str = 'fp16',
        device: Optional[torch.device] = None,
        device_map: Optional[Union[str, List[int]]] = None,
        eval_mode: bool = True,
        **kwargs,
    ):
        """Create a model of the given name."""

        super().__init__()

        self._model_name_or_path = model_name_or_path
        self._adapter_name_or_path = adapter_name_or_path

        self._precision = precision
        self._dtype, self._device = auto_dtype_and_device(precision, device)

        self._device_map = device_map or get_device_map(self._device)

        self._eval_mode = eval_mode

        self.load_model_and_transforms(
            model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )

        # turn the eval mode off `eval_mode=False` in training
        if self._eval_mode:
            self.model.eval()

        self.post_init(eval_mode=eval_mode, **kwargs)

    def load_model_and_transforms(
        self,
        model_name_or_path: str,
        adapter_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
    ):
        from .loading import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            precision=self._precision,
            dtype=self._dtype,
            device=self._device,
            device_map=self._device_map,
            use_fast=False,
        )

        if adapter_name_or_path is not None:
            self.load_adapter(adapter_name_or_path)

    def post_init(self, eval_mode: bool = True, **kwargs):

        if kwargs.get('use_compiled', False):

            import sys

            if torch.__version__ >= "2" and sys.platform != "win32":
                # TODO: use hidet backend in future: https://github.com/hidet-org/hidet
                self.model.forward = torch.compile(
                    self.model.forward,
                    dynamic=True,
                )
            else:
                self.model = torch.jit.script(self.model)
                if eval_mode:
                    self.model = torch.jit.optimize_for_inference(self.model)

    def load_adapter(self, adapter_name_or_path: str):
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_name_or_path,
            device_map={'': self._device or 0}
            if (self._device_map is None)
            else self._device_map,
        )

    def create_prompt_for_chat(self, messages: List[dict]) -> str:
        """Convert messages to a prompt string."""
        string_messages = []
        for message in messages:
            role = message['role']
            content = message['content']
            string_message = f"{role}: {content}"

            string_messages.append(string_message)

        string_messages.append(f"assistant: ")
        return "\n".join(string_messages)
