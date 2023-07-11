from typing import List, Optional, Union

import torch

from open_gpt.models.llama.modeling import LlamaModel


class VicunaModel(LlamaModel):
    """Wrapper for Vicuna model, which is a fine-tuned LLaMA model.

    Vicuna is trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT.
    See https://vicuna.lmsys.org/ for more details.

    The quick way to use Vicuna via :meth:`open_gpt.create_model`:

    ```python
    import open_gpt

    model = open_gpt.create_model('lmsys/vicuna-7b-delta-v1.1')

    # Generate text
    text_out = model.generate_text(prompts='Hello, my name is', max_length=50)
    ```

    If you want to run inference with lower precision and/or on a specific device, you can do:

    ```python
    import open_gpt

    model = open_gpt.create_model(
        'lmsys/vicuna-7b-delta-v1.1', precision='fp16', device_map='balanced'
    )
    ```
    """

    def load_model_and_transforms(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        **kwargs
    ):
        # Difference between different versions of Vicuna
        # See https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md
        if 'delta' in model_name_or_path:
            from .loading import load_model_and_tokenizer

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                dtype=self._dtype,
                precision=self._precision,
                device=self._device,
                device_map=self._device_map,
            )
        else:
            super(LlamaModel, self).load_model_and_transforms(
                model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                **kwargs
            )
