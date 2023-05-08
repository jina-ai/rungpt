from typing import Optional

from open_gpt.models.modeling import BaseModel


class PythiaModel(BaseModel):
    no_split_module_classes = ["GPTNeoXLayer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
