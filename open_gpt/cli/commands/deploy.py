from cleo.commands.command import Command
from cleo.helpers import argument, option


class DeployCommand(Command):
    name = "deploy"

    description = "Start to deploy a model to JCloud or AWS."

    arguments = [argument("model_name", "The name of the model to serve.")]
    options = [
        option(
            'grpc_port',
            None,
            'The gRPC port to serve the model on.',
            flag=False,
            default=51001,
        ),
        option(
            'http_port',
            None,
            'The HTTP port to serve the model on.',
            flag=False,
            default=51002,
        ),
        option(
            'cloud',
            None,
            'The cloud to deploy the model on.',
            flag=False,
            default='jcloud',
        ),
        option('enable_cors', None, 'Enable CORS.', flag=True),
        option(
            'precision', None, 'The precision of the model.', flag=False, default='fp16'
        ),
        option(
            'adapter_name_or_path', None, 'The name or path of the adapter checkpoint.'
        ),
        option(
            'device_map',
            None,
            'The device map of the model.',
            flag=False,
            default='balanced',
        ),
        option(
            "replicas", "r", "The number of replicas to serve.", flag=False, default=1
        ),
        option(
            "instance_type",
            None,
            "The instance used to deploy model.",
            flag=False,
            default='G3',
        ),
        option("config", None, "The config YAML used to deploy.", flag=False),
    ]

    help = """\
    This command allows you to deploy a model on JCloud or AWS.

    To start a model deploying, you can run:

        <comment>opengpt deploy facebook/llama-7b</comment>"""

    def handle(self) -> int:
        if self.option('cloud') == 'jcloud':
            from open_gpt.config import settings
            from open_gpt.serve.flow import SimpleFlow

            if self.option('config') is None:
                flow = SimpleFlow(
                    name=self.argument('model_name'),
                    template='flow.yml.jinja2',
                    executor_params=self._build_executor_params(),
                    gateway_params=self._build_gateway_params(),
                    labels={'app': 'opengpt'},
                    replicas=self.option('replicas'),
                    jina_version=settings.jina_version,
                    instance_type=self.option('instance_type'),
                )
                self.asyncify(flow.deploy(dry_run=True))
            else:
                raise NotImplementedError()

        elif self.option('cloud') == 'aws':
            raise NotImplementedError()
        return 0

    def _build_executor_params(self):
        return {
            'model_name_or_path': self.argument('model_name'),
            'precision': self.option('precision'),
            'adapter_name_or_path': self.option('adapter_name_or_path'),
            'device_map': self.option('device_map'),
        }

    def _build_gateway_params(self):
        return {
            'grpc_port': self.option('grpc_port'),
            'http_port': self.option('http_port'),
            'cors': self.option('enable_cors'),
        }

    def asyncify(self, f):
        import asyncio
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(f(*args, **kwargs))

        return wrapper
