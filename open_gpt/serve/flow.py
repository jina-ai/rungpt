import tempfile
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from jcloud.flow import CloudFlow
from jinja2 import BaseLoader, Environment, FileSystemLoader

from open_gpt.config import settings
from open_gpt.serve.helper import get_jina_latest_version


class SimpleFlow:
    def __init__(
        self,
        name: str,
        template: str,
        jina_version: str,
        executor_params: dict = {},
        gateway_params: dict = {},
        http_port: str = '51000',
        grpc_port: str = '52000',
        replicas: int = 1,
        instance_type: str = 'G3',
        **kwargs,
    ):
        self.name = name
        self.template = template
        self.executor_params = executor_params

        self.replicas = replicas
        self.jina_version = jina_version or get_jina_latest_version()

        self.instance_type = instance_type

        self.http_port = http_port
        self.grpc_port = grpc_port

        self.gateway_params = gateway_params
        self.gateway_image = (
            f'docker://{settings.gateway_name}:{settings.gateway_version}'
        )

    @cached_property
    def jinja_template(self):
        if self.template.endswith('.jinja2'):
            env = Environment(loader=FileSystemLoader(self.template))
            return env.get_template(self.template)
        else:
            return Environment(loader=BaseLoader()).from_string(self.template)

    def create_flow(
        self,
        labels: Optional[dict] = None,
        executor_params: Optional[dict] = None,
        output_path: Optional[Union[str, 'Path']] = None,
    ):
        executor_params = executor_params or {}
        executor_params.update(self.executor_params)

        content = self.jinja_template.render(
            deployment_name=self.name,
            gateway_image=self.gateway_image,
            gateway_params=self.gateway_params,
            http_port=self.http_port,
            grpc_port=self.grpc_port,
            replicas=self.replicas,
            executor_params=executor_params,
            jina_version=self.jina_version,
            labels=labels,
            instance_type=self.instance_tpye,
        )
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            return output_path
        else:
            return content

    def deploy(self, dry_run: bool = False):
        with tempfile.NamedTemporaryFile() as f:
            flow_path = self.create_flow(output_path=None)
            if not dry_run:
                return CloudFlow(path=flow_path)._deploy()
            else:
                return flow_path
