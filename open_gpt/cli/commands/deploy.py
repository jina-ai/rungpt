from cleo.commands.command import Command
from cleo.helpers import argument, option


class DeployCommand(Command):
    name = "deploy"

    description = "Start to deploy a model to cloud."

    arguments = [argument("model_name", "The name of the model to serve.")]
    options = [
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
            default='G2',
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

        <comment>opengpt deploy stabilityai/stablelm-tuned-alpha-3b</comment>"""

    def handle(self) -> int:
        if self.option('cloud') == 'jina':
            from open_gpt.factory import create_flow
            from open_gpt.helper import asyncify
            from open_gpt.serve.flow import deploy

            if self.option('config') is None:
                flow_yaml = create_flow(
                    model_name_or_path=self.argument('model_name'),
                    cors=self.option('enable_cors'),
                    uses_with={
                        'precision': self.option('precision'),
                        'adapter_name_or_path': self.option('adapter_name_or_path'),
                        'device_map': self.option('device_map'),
                    },
                    replicas=self.option('replicas'),
                    instance_type=self.option('instance_type'),
                    dockerized=True,
                    return_yaml=True,
                )

                if self.option('dry_run'):
                    self.line(f"{flow_yaml}")
                else:
                    asyncify(deploy)(flow=flow_yaml)
            else:
                raise NotImplementedError(
                    'Deploying with customized config is not supported yet.'
                )

        else:
            raise ValueError(
                f'Cloud {self.option("cloud")} is not supported yet. You can try to deploy on Jina instead.'
            )
        return 0
