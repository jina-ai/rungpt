from pydantic import BaseSettings

# jina core
DEFAULT_JINA_VERSION = ''

# gateway
DEFAULT_GATEWAY_NAME = 'inferenceteam/paywall-gateway'
DEFAULT_GATEWAY_VERSION = 'latest'


class Settings(BaseSettings):

    gateway_name: str = DEFAULT_GATEWAY_NAME
    gateway_version: str = DEFAULT_GATEWAY_VERSION

    jina_version: str = DEFAULT_JINA_VERSION


settings = Settings()


def _load_yaml_config(fname: str) -> dict:
    import yaml

    with open(fname, 'r') as f:
        return yaml.safe_load(f)
