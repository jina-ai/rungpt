from cleo.commands.command import Command


class AboutCommand(Command):
    name = "about"

    description = "Shows information about RunGPT."

    def handle(self) -> int:
        from open_gpt import __version__

        self.line(
            f"""\
<info>RunGPT - An open-source cloud-native model serving framework.</info>

Version: {__version__}</info>

<comment>RunGPT is a open-source cloud-native model serving framework\
 and libraries.
See <fg=blue>https://github.com/jina-ai/RunGPT</> for more information.</comment>\
"""
        )

        return 0
