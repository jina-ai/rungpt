from cleo.commands.command import Command
from cleo.helpers import argument, option


class DeployCommand(Command):
    name = "deploy"

    description = "Start to deploy a model to cloud."

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
            'name',
            None,
            'The name of the deployment.',
            flag=False,
            default='open-gpt-deployment',
        ),
        option(
            'cloud',
            None,
            'The cloud to deploy the model on.',
            flag=False,
            default='jina',
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
        option(
            'dry_run',
            None,
            'Whether to run the deployment in dry run mode.',
            flag=True,
        ),
        option("config", "c", "The config YAML used to deploy.", flag=False),
    ]

    help = """\
    This command allows you to deploy a model on cloud.

    To start a model deploying, you can run:

        <comment>opengpt deploy facebook/llama-7b</comment>"""

    def handle(self) -> int:
        if self.option('cloud') == 'jina':
            from open_gpt.factory import create_flow
            from open_gpt.serve.flow import deploy

            if self.option('config') is None:
                flow = create_flow(
                    self.argument('model_name'),
                    grpc_port=self.option('grpc_port'),
                    http_port=self.option('http_port'),
                    cors=self.option('enable_cors'),
                    uses_with={
                        'precision': self.option('precision'),
                        'adapter_name_or_path': self.option('adapter_name_or_path'),
                        'device_map': self.option('device_map'),
                    },
                    replicas=self.option('replicas'),
                    instance_type=self.option('instance_type'),
                    return_yaml=True,
                )
                deploy(flow, dry_run=self.option('dry_run'))
            else:
                raise NotImplementedError(
                    'Deploying with customized config is not supported yet.'
                )

        elif self.option('cloud') == 'aws':
            raise NotImplementedError('Deploying on AWS is not supported yet.')
        return 0

    @staticmethod
    def asyncify(f):
        import asyncio
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(f(*args, **kwargs))

        return wrapper
