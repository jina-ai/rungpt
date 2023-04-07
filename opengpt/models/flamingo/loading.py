import open_clip
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance

from ..hf_model import load_model_and_tokenizer
from .modeling import FlamingoModel


def load_model_and_transforms(
    clip_model_name: str,
    lang_model_name_or_path: str,
    tokenizer_name_or_path: str,
    decoder_layers_attr_name: str = None,
    **kwargs,
):
    """Load a Flamingo model and its associated image and text processors.

    :param clip_model_name: The name of the CLIP model to use.
    :param lang_model_name_or_path: The name or path of the language model to use.
    :param tokenizer_name_or_path: The name or path of the tokenizer to use.
    :param decoder_layers_attr_name: The name of the attribute that specifies the decoder layers.
    """

    # load the vision model
    model_name, *pretrained = clip_model_name.split("::")
    pretrained = pretrained[0] if len(pretrained) == 1 else 'openai'
    clip_model, _, image_processor = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    # set the vision encoder to output the visual features
    clip_model.visual.output_tokens = True

    # remove text encoder to save footprint and memory
    if hasattr(clip_model, 'text'):
        del clip_model.text
    elif hasattr(clip_model, 'transformer'):
        del clip_model.transformer

    # load the language model
    lang_model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=lang_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )

    # add Flamingo special tokens to the tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    extend_instance(lang_model, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_model)
    lang_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_model.resize_token_embeddings(len(tokenizer))

    flamingo_config = {
        "image_size": open_clip.get_model_config(model_name)["vision_cfg"]["width"],
        "cross_attn_every_n_layers": 1,
        "end_chunk_token_id": tokenizer.encode("<|endofchunk|>")[-1],
        "media_token_id": tokenizer.encode("<image>")[-1],
    }

    model = FlamingoModel(
        clip_model,
        lang_model,
        model_config=flamingo_config,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.language_model.gated_cross_attn_layers.requires_grad_(True)
    # TODO: only unfreeze the input embeddings of the additional special tokens
    model.language_model.get_input_embeddings().requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, tokenizer


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
