import os.path
from pathlib import Path
from typing import List, Optional, Union

import torch


def create_model(
    model_name: str,
    precision: str = 'fp16',
    adapter_name_or_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    device_map: Optional[Union[str, List[int]]] = None,
    **kwargs,
):
    """Create a model.

    :param model_name: The name of the model to create.
    :param adapter_name_or_path: The name or path of the adapter to use for the model.
            This is only used for models that support adapters for fine-tuning. Defaults to ``None``.
    :param precision: The precision to use for the model. Can be one of ``"fp16"``, ``"fp32"``, ``"bit8"`` or ``"bit4"``. Defaults to ``"fp16"``.
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
            adapter_name_or_path=adapter_name_or_path,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('lmsys/vicuna') or model_name.startswith(
        'CarperAI/stable-vicuna'
    ):
        assert not model_name.endswith(
            'v0'
        ), 'You are using an outdated model, please use the newer version ``v1.1+``'
        from .models.vicuna.modeling import VicunaModel

        assert adapter_name_or_path is None, 'Vicuna does not support adapter'

        return VicunaModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('EleutherAI/pythia'):
        from .models.pythia.modeling import PythiaModel

        assert adapter_name_or_path is None, 'Pythia does not support adapter'

        return PythiaModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('stabilityai/stablelm'):
        from .models.stablelm.modeling import StableLMModel

        assert adapter_name_or_path is None, 'StableLM does not support adapter'

        return StableLMModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('fnlp/moss'):
        from .models.moss.modeling import MossModel

        assert adapter_name_or_path is None, 'Moss does not support adapter'

        return MossModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('openflamingo/OpenFlamingo'):
        from .models.flamingo.modeling import FlamingoModel

        assert adapter_name_or_path is None, 'Flamingo does not support adapter'

        return FlamingoModel(
            model_name,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )
    elif model_name.startswith('sgugger/rwkv') or model_name.startswith(
        'ybelkada/rwkv'
    ):
        from .models.rwkv.modeling import RWKVModel

        return RWKVModel(
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
            adapter_name_or_path=adapter_name_or_path,
            device=device,
            precision=precision,
            device_map=device_map,
            **kwargs,
        )


def create_flow(
    model_name_or_path: str,
    grpc_port: int = 51001,
    http_port: int = 51002,
    cors: bool = False,
    uses_with: dict = {},
    replicas: int = 1,
    instance_type: Optional[str] = None,
    dockerized: bool = False,
    return_yaml: bool = True,
):
    from jina import Flow

    from open_gpt import __jina_version__, __version__
    from open_gpt.serve.flow import get_template

    # normalize the model name to be used as flow executor name
    norm_name = model_name_or_path.split('/')[-1]
    norm_name = norm_name.replace('-', '_').replace('.', '_').lower()

    # HOTFIX: patch to avoid to use pre-release version
    __VERSION_TAG__ = f'v{__version__}'
    if 'dev' in __version__:
        __VERSION_TAG__ = 'latest'

    deployment_params = {
        'deployment_name': f'{norm_name}',
        'http_port': http_port,
        'grpc_port': grpc_port,
        'executor_params': {
            'model_name_or_path': model_name_or_path,
            'adapter_name_or_path': uses_with.get('adapter_name_or_path') or '',
            'precision': uses_with.get('precision', 'fp16'),
            'device_map': uses_with.get('device_map', 'balanced'),
        },
        'gateway_params': {'cors': cors},
        'jina_version': __jina_version__,
        'replicas': replicas,
        'labels': {'app': 'open_gpt', 'version': __VERSION_TAG__},
    }

    yaml = get_template('flow.yml.jinja2').render(
        dockerized=dockerized,
        gateway_image=f'docker://jinaai/open_gpt_gateway:{__VERSION_TAG__}',
        gateway_module='Gateway',
        executor_image=f'docker://jinaai/open_gpt_executor:{__VERSION_TAG__}',
        executor_module='CausualLMExecutor'
        if 'flamingo' not in model_name_or_path
        else 'FlamingoExecutor',
        instance_type=instance_type,
        **deployment_params,
    )

    if return_yaml:
        return yaml
    else:
        return Flow.load_config(yaml)
