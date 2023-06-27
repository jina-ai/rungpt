from importlib import import_module
from typing import Callable

from cleo.application import Application as BaseApplication
from cleo.commands.command import Command

from open_gpt import __version__
from open_gpt.cli.command_loader import CommandLoader


def load_command(name: str) -> Callable[[], Command]:
    def _load() -> Command:
        words = name.split(" ")
        module = import_module("open_gpt.cli.commands." + ".".join(words))
        command_class = getattr(module, "".join(c.title() for c in words) + "Command")
        command: Command = command_class()
        return command

    return _load


COMMANDS = [
    "about",
    "serve",
    "deploy",
    "playground",
    "quantize",
]


class Application(BaseApplication):
    def __init__(self) -> None:
        super().__init__("opengpt", __version__)

        command_loader = CommandLoader({name: load_command(name) for name in COMMANDS})
        self.set_command_loader(command_loader)


def main() -> int:
    exit_code: int = Application().run()
    return exit_code


if __name__ == "__main__":
    main()
