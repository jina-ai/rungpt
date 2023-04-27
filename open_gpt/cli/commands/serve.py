from cleo.commands.command import Command
from cleo.helpers import argument, option


class ServeCommand(Command):
    name = "serve"

    description = "Start a model serving locally."

    arguments = [argument("model_name", "The name of the model to serve.")]
    options = [
        option(
            "protocol",
            None,
            "The protocol to serve the model on.",
            flag=False,
            default="grpc",
        ),
        option(
            "port", "p", "The port to serve the model on.", flag=False, default=51000
        ),
        option(
            "replicas", "r", "The number of replicas to serve.", flag=False, default=1
        ),
    ]

    help = """\
    This command allows you to start a model serving locally.
    
    To start a model serving locally, you can run:
        
        <comment>opengpt serve facebook/llama-7b</comment>"""

    def handle(self) -> int:
        from open_gpt.serve.flow import create_flow

        with create_flow(
            self.argument('model_name'),
            protocol=self.option('protocol'),
            port=self.option('port'),
            replicas=self.option('replicas'),
        ) as flow:
            flow.block()

        return 0
