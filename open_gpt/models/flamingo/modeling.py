from typing import Callable, Optional, Union

import torch
from einops import rearrange
from open_flamingo.src.helpers import PerceiverResampler
from torch import nn


class FlamingoModel(nn.Module):
    def __init__(
        self,
        vision_encoder: 'nn.Module',
        language_model: 'nn.Module',
        model_config: dict = {},
        device: Optional[Union[str, 'torch.device']] = 'cpu',
        **kwargs,
    ):
        """An open source version of DeepMind's Flamingo model!
        Adapted from the original implementation at https://github.com/mlfoundations/open_flamingo

        :param vision_encoder: the vision encoder to extract visual features, e.g. CLIP model
        :param language_model: the language model to extract textual features and generate the output texts, e.g., LLaMa model
        :param model_config: a dictionary of model configuration
        :param device: the device to run the model on
        :param kwargs: other arguments
        """
        super().__init__()

        self.model_config = model_config

        self.vision_encoder = vision_encoder
        self.language_model = language_model

        self.perceiver = PerceiverResampler(dim=self.model_config['image_size'])
        self.perceiver.to(device)

        self.media_token_id = model_config['media_token_id']
        self.end_chunk_token_id = model_config['end_chunk_token_id']

        self.language_model.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=model_config['image_size'],
            cross_attn_every_n_layers=model_config['cross_attn_every_n_layers'],
            use_media_placement_augmentation=False,
        )
        self.language_model.gated_cross_attn_layers.to(device)

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

        output = self.language_model(
            input_ids=text_inputs,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=False,
        )

        self.language_model.clear_conditioned_layers()

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
        early_stopping: bool = False,
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
        if num_beams > 1:
            vision_inputs = vision_inputs.repeat_interleave(num_beams, dim=0)

        vision_x = self._vision_encode(vision_inputs=vision_inputs)

        print(f'====> encode vision done {vision_x.device}...')

        if not self.language_model.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        media_locations = text_inputs == self.media_token_id
        attend_previous = False

        for layer in self.language_model.get_decoder().layers:
            layer.condition_media_locations(media_locations)
            layer.condition_attend_previous(attend_previous)

        print(f'===> start generation ...')

        output = self.language_model.generate(
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

        self.language_model.clear_conditioned_layers()
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

        device = next(iter(self.vision_encoder.parameters())).device
        vision_inputs = vision_inputs.to(device)

        vision_x = rearrange(vision_inputs, "B T F c h w -> (B T F) c h w")

        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]

        vision_x = rearrange(vision_x, "(B T F) v d -> B T F v d", B=B, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (B, T, n, d)

        device = next(iter(self.language_model.parameters())).device

        vision_x = vision_x.to(device)

        for layer in self.language_model._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
