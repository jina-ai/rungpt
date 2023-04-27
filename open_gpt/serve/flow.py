from jina import Flow

from ..logging import logger


def create_flow(
    model_name_or_path: str, protocol='http', port=51000, replicas: int = 1
):
    from .executors import CausualLMExecutor

    return Flow(protocol=protocol, port=port, cors=True).add(
        uses=CausualLMExecutor,
        uses_with={'model_name_or_path': model_name_or_path},
        name='causal_lm',
        replicas=replicas,
        timeout_ready=-1,
    )
