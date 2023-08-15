import random
from typing import Optional, Union

import torch
import torch.nn as nn
from open_flamingo.src.helpers import GatedCrossAttentionBlock
from open_flamingo.src.utils import getattr_recursive, setattr_recursive

from ...helper import auto_dtype_and_device


class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

        self._device = self.decoder_layer.parameters().__next__().device
        self._dtype = self.decoder_layer.parameters().__next__().dtype

        if self._dtype not in [torch.float16, torch.float32]:
            self._dtype = torch.float16

        # This is a hack to guarantee that the gated_cross_attn_layer is on the same device as the decoder_layer
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.to(self._device).to(self._dtype)

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(
                lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
            )

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id: int,
        vis_hidden_size: int,
        cross_attn_every_n_layers: int,
        use_media_placement_augmentation: bool,
        only_attend_previous: bool,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                    # only_attend_previous=only_attend_previous,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self._get_decoder_layers()
                    )
                ]
            )
        )
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.only_attend_previous = only_attend_previous
        self.initialized_flamingo = True

    def call_forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        # IMPORTANT: Force `attend_previous` to True when we place training data as <image>caption<|endofchunk|>
        # attend_previous = (
        #     (random.random() < 0.5) if self.use_media_placement_augmentation else False
        # )
        attend_previous = self.only_attend_previous

        for layer in self.get_decoder().layers:
            # print(f'===> layer: {layer}')
            layer.condition_media_locations(media_locations)
            layer.condition_attend_previous(attend_previous)

        return super().forward(
            *input, **kwargs
        )  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_attend_previous(None)
