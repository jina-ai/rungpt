from typing import Callable, Optional, Union

import torch
from einops import rearrange
from open_flamingo.src.helpers import PerceiverResampler
from torch import nn

from open_gpt.logs import logger

from ...helper import auto_dtype_and_device


class FlamingoLMModel(nn.Module):

    base_model_prefix = "flamingo"
    _no_split_modules = ["FlamingoPerceiverBlock", "CLIPEncoderLayer", "FlamingoLayer"]

    def __init__(
        self,
        vision_encoder: 'nn.Module',
        language_model: 'nn.Module',
        device: Optional[Union[str, 'torch.device']] = None,
        dtype: Optional[Union[str, 'torch.dtype']] = None,
        model_config: dict = {},
        **kwargs,
    ):
        """An open source version of DeepMind's Flamingo model!
        Adapted from the original implementation at https://github.com/mlfoundations/open_flamingo

        :param vision_encoder: the vision encoder to extract visual features, e.g. CLIP model
        :param language_model: the language model to extract textual features and generate the output texts, e.g., LLaMa model
        :param model_config: a dictionary of model configuration
        :param device: the device to run the model on
        :param dtype: the data type to run the model on
        :param kwargs: other arguments
        """

        super().__init__()

        self._dtype, self._device = auto_dtype_and_device(dtype, device)

        self.model_config = model_config
        self.vision_encoder = vision_encoder
        self.lang_encoder = language_model

        self.perceiver = PerceiverResampler(dim=self.model_config['image_size'])
        if str(self._dtype) == 'torch.float16':
            self.perceiver.half()
        self.perceiver.to(self._device)

        self.media_token_id = model_config['media_token_id']
        self.end_chunk_token_id = model_config['end_chunk_token_id']

        logger.debug(
            f"media_token_id: {self.media_token_id}, end_chunk_token_id: {self.end_chunk_token_id}"
        )

        self.lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=model_config['image_size'],
            cross_attn_every_n_layers=model_config['cross_attn_every_n_layers'],
            use_media_placement_augmentation=True,
            only_attend_previous=True,
        )

        if hasattr(self.lang_encoder, "_hf_hook"):
            # logger.debug(f'lang_encoder has _hf_hook, {self.lang_encoder._hf_hook}')
            import functools

            from accelerate.hooks import add_hook_to_module, remove_hook_from_module

            hook = self.lang_encoder._hf_hook

            remove_hook_from_module(self.lang_encoder)

            old_forward = self.lang_encoder.call_forward

            self.lang_encoder = hook.init_hook(self.lang_encoder)
            self.lang_encoder._hf_hook = hook

            @functools.wraps(old_forward)
            def new_forward(*args, **kwargs):
                args, kwargs = self.lang_encoder._hf_hook.pre_forward(
                    self.lang_encoder, *args, **kwargs
                )
                if self.lang_encoder._hf_hook.no_grad:
                    with torch.no_grad():
                        output = old_forward(*args, **kwargs)
                else:
                    output = old_forward(*args, **kwargs)
                return self.lang_encoder._hf_hook.post_forward(
                    self.lang_encoder, output
                )

            self.lang_encoder.forward = new_forward

            # add_hook_to_module(self.lang_encoder, old_hook)

    def forward(
        self,
        vision_inputs: 'torch.Tensor',
        text_inputs: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
        labels: Optional['torch.Tensor'] = None,
        past_key_values: Optional['torch.Tensor'] = None,
    ):
        """

        :param vision_inputs: vision images input with shape (B, T, F, Channels, Height, Width).
                Images in the same chunk are collated along T, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        :param text_inputs: language text input with shape (batch_size, sequence_length)
        :param attention_mask: attention mask, 1 for tokens that are not masked, 0 for masked tokens. Defaults to None.
        :param labels: labels for computing the language modeling loss. Defaults to None.
        :param past_key_values: cached past key and values for fast decoding. Defaults to None.
        """

        self._vision_encode(vision_inputs)

        output = self.lang_encoder(
            input_ids=text_inputs,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=False,
        )

        self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_inputs: 'torch.Tensor',
        text_inputs: 'torch.Tensor',
        attention_mask: torch.Tensor = None,
        num_beams: int = 1,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        no_repeat_ngram_size: int = 0,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        early_stopping: bool = True,
    ):
        """
        Generate text conditioned on vision and language inputs.

        :param vision_inputs: vision images input with shape (B, T, F, Channels, Height, Width).
                Images in the same chunk are collated along T, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        :param text_inputs: language text input with shape (batch_size, sequence_length)
        :param attention_mask: attention mask, 1 for tokens that are not masked, 0 for masked tokens. Defaults to None.
        :param num_beams: number of beams for beam search. Defaults to 1.
        :param max_new_tokens: maximum number of new tokens to generate. Defaults to None.
        :param temperature: temperature for sampling. Defaults to 1.0.
        :param top_k: top k for sampling. Defaults to 0.
        :param top_p: top p for sampling. Defaults to 1.0.
        :param no_repeat_ngram_size: no repeat ngram size. Defaults to 0.
        :param prefix_allowed_tokens_fn: prefix allowed tokens function. Defaults to None.
        :param length_penalty: length penalty. Defaults to 1.0.
        :param num_return_sequences: number of return sequences. Defaults to 1.
        :param do_sample: whether to sample or not. Defaults to False.
        :param early_stopping: whether to stop early or not. Defaults to False.

        :return: text_inputs with generated tokens appended to it (batch_size, sequence_length)
        """
        vision_inputs = vision_inputs.to(dtype=self._dtype, device=self._device)
        text_inputs = text_inputs.to(self._device)

        if num_beams > 1:
            vision_inputs = vision_inputs.repeat_interleave(num_beams, dim=0)

        _vision_x = self._vision_encode(vision_inputs=vision_inputs)

        if not self.lang_encoder.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        output = self.lang_encoder.generate(
            text_inputs,
            attention_mask=attention_mask,
            eos_token_id=self.end_chunk_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output

    def _vision_encode(self, vision_inputs: 'torch.Tensor') -> 'torch.Tensor':
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.

        :param vision_inputs: vision images input with shape (B, T, F, channels, height, width).
                Images in the same chunk are collated along T, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        :return: media tokens with shape (batch_size, frames, tokens, dim)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert (
            vision_inputs.ndim == 6
        ), "vision_inputs should be of shape (B, T, F, Channels, Height, Width)"
        B, T, F = vision_inputs.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_inputs, "B T F c h w -> (B T F) c h w")

        with torch.no_grad():
            vision_x = self.vision_encoder.visual(
                vision_x.to(self._dtype).to(self._device)
            )[1]

        vision_x = rearrange(vision_x, "(B T F) v d -> B T F v d", B=B, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (B, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
