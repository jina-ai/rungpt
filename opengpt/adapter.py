"""The model adapter for OpenGPT models.

The model adapter is a wrapper around the model that provides a light-weight way to
enrich the model with additional functionality.
"""
import torch
from torch import nn


class ModelAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, adapter_configs: dict, **kwargs):
        super().__init__()
        self._base_model = base_model
        self._adapter_configs = adapter_configs

    def forward(self):
        ...
