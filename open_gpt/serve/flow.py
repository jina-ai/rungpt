from jina import Flow


def create_flow(protocol='http', port=51000, replicas: int = 1):
    from .executors import CausualLMExecutor

    return Flow(protocol=protocol, port=port, cors=True).add(
        uses=CausualLMExecutor, name='causal_lm', replicas=replicas
    )
