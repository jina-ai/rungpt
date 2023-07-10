from functools import cache

from jcloud.flow import CloudFlow
from jinja2 import BaseLoader, Environment, FileSystemLoader


@cache
def get_template(template):
    if template.endswith('.jinja2'):
        from open_gpt import __resources_path__

        env = Environment(loader=FileSystemLoader(__resources_path__))
        return env.get_template(template)
    else:
        return Environment(loader=BaseLoader()).from_string(template)


async def deploy(flow: str):
    import os
    import tempfile

    if os.path.isfile(flow):
        return await CloudFlow(path=flow).deploy()
    else:
        with tempfile.NamedTemporaryFile() as f:
            with open(f.name, 'w') as _:
                _.write(flow)
            return await CloudFlow(path=f.name).__aenter__()
