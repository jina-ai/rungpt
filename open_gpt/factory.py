from pathlib import Path
from typing import List, Optional, Union

import torch


def create_model(
    model_name: str,
    precision: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs,
):
    """Create a model.

    :param model_name: The name of the model to create.
    :param precision: The precision to use. Can be one of ``"float16"``, ``"float32"``, ``"float64"``, ``"bfloat16"``, ``"mixed"`` or ``None``.
    :param device: The device to use. Can be one of ``"cpu"``, ``"cuda"``, ``"cuda:X"`` or ``None``.
    :param device_map: The device map to use. Can be one of ``"balanced"``, ``"single"`` or a list of device IDs.
    :param kwargs: Additional keyword arguments to pass to the model.
    """

    if model_name.startswith('facebook/llama') or model_name.startswith(
        'decapoda-research/llama'
    ):
        from .models.llama.modeling import LlamaModel

        return LlamaModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('lmsys/vicuna'):
        assert not model_name.endswith(
            'v0'
        ), 'You are using an outdated model, please use the newer version ``v1.1+``'
        from .models.vicuna.modeling import VicunaModel

        return VicunaModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('EleutherAI/pythia'):
        from .models.pythia.modeling import PythiaModel

        return PythiaModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('stabilityai/stablelm'):
        from .models.stablelm.modeling import StableLMModel

        return StableLMModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('fnlp/moss'):
        from .models.moss.modeling import MossModel

        return MossModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('openflamingo/OpenFlamingo'):
        from .models.flamingo.modeling import FlamingoModel

        return FlamingoModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    else:
        from .models.modeling import BaseModel

        return BaseModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )


def create_flow(
    model_name_or_path: str, protocol='http', port=51000, replicas: int = 1
):
    from jina import Flow

    if 'flamingo' in model_name_or_path:
        from .serve.executors.flamingo import FlamingoExecutor as Executor
    else:
        from serve.executors import CausualLMExecutor as Executor

    # normalize the model name to be used as flow executor name
    norm_name = model_name_or_path.split('/')[-1]
    norm_name = norm_name.replace('-', '_').lower()

    return Flow(protocol=protocol, port=port, cors=True).add(
        uses=Executor,
        uses_with={'model_name_or_path': model_name_or_path},
        name=f'{norm_name}_executor',
        replicas=replicas,
        timeout_ready=-1,
    )
