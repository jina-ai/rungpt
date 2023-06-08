from open_gpt.models.modeling import BaseModel


class PythiaModel(BaseModel):
    """Pythia model.

    It contains two sets of eight models of sizes 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B. For each size,
    there are two models: one trained on the Pile, and one trained on the Pile after the dataset has been globally deduplicated.

    See https://github.com/EleutherAI/pythia for more information.

    The quick way to use Pythia via :meth:`open_gpt.create_model`:

    ```python
    import open_gpt

    model = open_gpt.create_model(
        'EleutherAI/pythia-12b-deduped', precision='fp16', device_map='balanced'
    )
    ```
    """
