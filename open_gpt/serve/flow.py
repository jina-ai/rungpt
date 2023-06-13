import tempfile
from functools import cached_property

from jcloud.flow import CloudFlow
from jinja2 import BaseLoader, Environment, FileSystemLoader


@cached_property
def jinja_template(template):
    if template.endswith('.jinja2'):
        from open_gpt.serve.helper import __resouce__

        env = Environment(loader=FileSystemLoader(__resouce__))
        return env.get_template(template)
    else:
        return Environment(loader=BaseLoader()).from_string(template)


def deploy(flow: str, dry_run: bool = False):
    with tempfile.NamedTemporaryFile() as f:
        with open(f.name, 'w') as _:
            _.write(flow)
        if not dry_run:
            return CloudFlow(path=f.name)._deploy()
        else:
            return f.name
