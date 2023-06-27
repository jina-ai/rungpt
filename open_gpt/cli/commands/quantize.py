from cleo.commands.command import Command
from cleo.helpers import argument, option


class QuantizeCommand(Command):
    name = 'quantize'

    description = 'quantize the model'

    arguments = [argument("model_name", "The name of the model to quantize.")]

    options = [
        option(
            'target',
            None,
            'The path to quantized checkpoint.',
            flag=False,
            default=''
        )
    ]

    def handle(self) -> int:
        from open_gpt.spqr import quantize
        _, _ = quantize(self.argument('model_name'), self.option('target'))
        return 0
