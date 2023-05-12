from cleo.commands.command import Command
from cleo.helpers import argument, option


class PlaygroundCommand(Command):
    name = "playground"

    description = "Start a online playground."

    # arguments = [argument("model_name", "The name of the model to serve.")]
    options = [
        option(
            "port", "p", "The port to serve the playground.", flag=False, default=52000
        ),
    ]

    help = """\
    This command allows you to start a online playground for a model.

    To start a online playground for a model, you can run:

        <comment>opengpt playground facebook/llama-7b</comment>"""

    def handle(self) -> int:
        from open_gpt.serve.playground.gradio import create_playground

        playground = create_playground()
        playground.queue(
            concurrency_count=8, status_update_rate=10, api_open=False
        ).launch(server_name='0.0.0.0', server_port=self.option('port'))
        return 0
