from typing import Callable

from cleo.commands.command import Command
from cleo.exceptions import CleoLogicError
from cleo.loaders.factory_command_loader import FactoryCommandLoader


class CommandLoader(FactoryCommandLoader):
    def register_factory(
        self, command_name: str, factory: Callable[[], Command]
    ) -> None:
        if command_name in self._factories:
            raise CleoLogicError(f'The command "{command_name}" already exists.')

        self._factories[command_name] = factory
