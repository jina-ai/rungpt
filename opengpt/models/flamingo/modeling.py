from typing import Optional

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
        **kwargs
    ):
        """An open source version of DeepMind's Flamingo model!
        Adapted from the original implementation at https://github.com/mlfoundations/open_flamingo

        :param vision_encoder: the vision encoder to extract visual features, e.g. CLIP model
        :param language_model: the language model to extract textual features and generate the output texts, e.g., LLaMa model
        :param model_config: a dictionary of model configuration
        :param kwargs: other arguments
        """
        super().__init__()

        self.model_config = model_config

        self.vision_encoder = vision_encoder
        self.language_model = language_model

        self.perceiver = PerceiverResampler(dim=self.model_config['image_size'])

        self.media_token_id = model_config['media_token_id']
        self.end_chunk_token_id = model_config['end_chunk_token_id']

        self.language_model.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=model_config['image_size'],
            cross_attn_every_n_layers=model_config['cross_attn_every_n_layers'],
            use_media_placement_augmentation=False,
        )

    def forward(
        self,
        vision_inputs: 'torch.Tensor',
        text_inputs: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
    ):
        """

        :param vision_inputs: vision images input with shape (B, T, F, Channels, Height, Width).
                Images in the same chunk are collated along T, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        :param text_inputs: language text input with shape (batch_size, sequence_length)
        :param attention_mask: attention mask, 1 for tokens that are not masked, 0 for masked tokens. Defaults to None.
        """

        self._vision_encode(vision_inputs)

        vision_inputs = rearrange(vision_inputs, 'b c h w -> b (h w) c')
        vision_inputs = self.perceiver(vision_inputs)

        # logits = self.language_model(
        #     vision_x=vision_x,
        #     lang_x=text_inputs,
        #     attention_mask=attention_mask,
        #     labels=labels,
        #     use_cached_vision_x=use_cached_vision_x,
        #     clear_conditioned_layers=clear_conditioned_layers,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        # )
        #
        # return logits

    def _vision_encode(self, vision_inputs: 'torch.Tensor'):
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
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(B T F) v d -> B T F v d", B=B, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (B, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
