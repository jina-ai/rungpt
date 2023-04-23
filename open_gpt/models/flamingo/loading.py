from typing import Optional, Union

import open_clip
import torch
from loguru import logger
from open_flamingo.src.utils import extend_instance


def load_model_and_transforms(
    model_name_or_path: str,
    vision_model_name_or_path: str,
    lang_model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    decoder_layers_attr_name: Optional[str] = None,
    device: 'torch.device' = torch.device('cuda'),
    dtype: 'torch.dtype' = torch.float16,
    **kwargs,
):
    """Load a Flamingo model and its associated image and text processors.

    :param model_name_or_path: The name or path of the model to load.
    :param vision_model_name_or_path: The name or path of the vision model to use.
    :param lang_model_name_or_path: The name or path of the language model to use.
    :param tokenizer_name_or_path: The name or path of the tokenizer to use. If not specified, the tokenizer associated with the model will be used.
    :param decoder_layers_attr_name: The name of the attribute that specifies the decoder layers.
    :param device: The device to load the model on.
    :param dtype: The dtype to load the model with.
    """

    from ...helper import cast_precision
    from ..llama.loading import (
        load_model_and_tokenizer as load_llama_model_and_tokenizer,
    )
    from .flamingo_lm import FlamingoLMMixin
    from .modeling import FlamingoModel

    # load the vision model
    precision = cast_precision(dtype)
    model_name, *pretrained = vision_model_name_or_path.split("::")
    pretrained = pretrained[0] if len(pretrained) == 1 else 'openai'
    clip_model, _, image_processor = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, precision=precision
    )
    # set the vision encoder to output the visual features
    clip_model.visual.output_tokens = True

    # remove text encoder to save footprint and memory
    if hasattr(clip_model, 'text'):
        del clip_model.text
    elif hasattr(clip_model, 'transformer'):
        del clip_model.transformer

    # execution_device = next(iter(clip_model.parameters())).device
    # add_hook_to_module(clip_model, AlignDevicesHook(io_same_device=True), append=True)
    #
    # attach_align_device_hook_on_blocks(
    #     clip_model,
    #     execution_device=execution_device,
    #     offload=None,
    #     offload_buffers=False,
    #     weights_map=None,
    #     preload_module_classes=None,
    # )

    # load the language model
    lang_model, tokenizer = load_llama_model_and_tokenizer(
        model_name_or_path=lang_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        device=device,
        dtype=dtype,
    )

    # add Flamingo special tokens to the tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # For generation padding tokens should be on the left
    tokenizer.padding_side = "left"

    extend_instance(lang_model, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_model)
    lang_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_model.resize_token_embeddings(len(tokenizer))

    flamingo_config = {
        "image_size": open_clip.get_model_config(model_name)["vision_cfg"]["width"],
        "cross_attn_every_n_layers": 4,
        "end_chunk_token_id": tokenizer.encode("<|endofchunk|>")[-1],
        "media_token_id": tokenizer.encode("<image>")[-1],
    }

    model = FlamingoModel(
        clip_model,
        lang_model,
        model_config=flamingo_config,
        device=device,
        dtype=dtype,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    # TODO: only unfreeze the input embeddings of the additional special tokens
    model.lang_encoder.get_input_embeddings().requires_grad_(True)

    logger.debug(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    # grab model checkpoint from huggingface hub
    import os

    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    assert (
        hf_token is not None
    ), "Please set HF_TOKEN environment variable to download model from HuggingFace Hub"

    checkpoint_path = hf_hub_download(
        model_name_or_path, "checkpoint.pt", token=hf_token
    )
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, tokenizer, image_processor


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
}
