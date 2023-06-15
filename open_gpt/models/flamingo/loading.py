from typing import List, Optional, Union

import open_clip
import torch
from open_flamingo.src.utils import extend_instance

from open_gpt.logs import logger


def load_model_and_transforms(
    model_name_or_path: str,
    vision_model_name_or_path: str,
    lang_model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    decoder_layers_attr_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    precision: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs,
):
    """Load a Flamingo model and its associated image and text processors.

    :param model_name_or_path: The name or path of the model to load.
    :param vision_model_name_or_path: The name or path of the vision model to use.
    :param lang_model_name_or_path: The name or path of the language model to use.
    :param tokenizer_name_or_path: The name or path of the tokenizer to use. If not specified, the tokenizer associated with the model will be used.
    :param decoder_layers_attr_name: The name of the attribute that specifies the decoder layers.
    :param precision: The precision to load the model with.
    :param device: The device to load the model on.
    :param dtype: The dtype to load the model with.
    """

    import os

    from ..llama.loading import (
        load_model_and_tokenizer as load_llama_model_and_tokenizer,
    )
    from .flamingo_lm import FlamingoLMMixin
    from .flamingo_model import FlamingoLMModel

    hf_token = os.environ.get("HF_TOKEN")
    assert (
        hf_token is not None
    ), "Please set HF_TOKEN environment variable to download model from HuggingFace Hub"

    assert (
        device_map is None
    ), f"`device_map={device_map}` is not supported for Flamingo models"

    # load the vision model
    model_name, *pretrained = vision_model_name_or_path.split("::")
    pretrained = pretrained[0] if len(pretrained) == 1 else 'openai'
    clip_model, _, image_processor = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, precision='fp16', device=device
    )
    # set the vision encoder to output the visual features
    clip_model.visual.output_tokens = True

    # remove text encoder to save footprint and memory
    if hasattr(clip_model, 'text'):
        del clip_model.text
    elif hasattr(clip_model, 'transformer'):
        del clip_model.transformer

    # patch to load llama models
    if lang_model_name_or_path.startswith('facebook/llama'):
        import os

        model_size = lang_model_name_or_path.split('-')[-1]
        if not os.path.exists(lang_model_name_or_path):
            lang_model_name_or_path = f"decapoda-research/llama-{model_size}-hf"

    # load the language model
    lang_model, tokenizer = load_llama_model_and_tokenizer(
        model_name_or_path=lang_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        precision=precision,
        device=device,
        dtype=dtype,
        device_map=device_map,
    )

    # add Flamingo special tokens to the tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    extend_instance(lang_model, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_model)
    lang_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_model.resize_token_embeddings(len(tokenizer))

    # flamingo_config = {
    #     "image_size": open_clip.get_model_config(model_name)["vision_cfg"]["width"],
    #     "cross_attn_every_n_layers": 4,
    #     "end_chunk_token_id": tokenizer.encode("<|endofchunk|>")[-1],
    #     "media_token_id": tokenizer.encode("<image>")[-1],
    # }

    flamingo_config = {
        "model_type": "flamingo",
        "image_size": open_clip.get_model_config(model_name)["vision_cfg"]["width"],
        "cross_attn_every_n_layers": 4,
        "end_chunk_token_id": tokenizer.encode("<|endofchunk|>")[-1],
        "media_token_id": tokenizer.encode("<image>")[-1],
        "tie_word_embeddings": False,
        "use_media_placement_augmentation": True,
        "only_attend_previous": True,
        "text_config": {"_name_or_path": "yahma/llama-7b-hf", "model_type": "llama"},
        "vision_config": {
            "_name_or_path": "openai/clip-vit-large-patch14",
            "model_type": "clip_vision_model",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "image_size": 224,
            "patch_size": 14,
        },
    }

    model = FlamingoLMModel(
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
    from huggingface_hub import hf_hub_download

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
