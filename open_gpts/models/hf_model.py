from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name_or_path: str, tokenizer_name_or_path: str):
    """Load a model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, local_files_only=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, local_files_only=False
    )
    return model, tokenizer
