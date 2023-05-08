from jina import Flow

from ..logging import logger


def create_flow(
    model_name_or_path: str, protocol='http', port=51000, replicas: int = 1
):
    from .executors import CausualLMExecutor

    # normalize the model name to be used as flow executor name
    norm_name = model_name_or_path.split('/')[-1]
    norm_name = norm_name.replace('-', '_')

    return Flow(protocol=protocol, port=port, cors=True).add(
        uses=CausualLMExecutor,
        uses_with={'model_name_or_path': model_name_or_path},
        name=f'{norm_name}_executor',
        replicas=replicas,
        timeout_ready=-1,
    )
