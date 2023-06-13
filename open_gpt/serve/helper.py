from pathlib import Path

__resource__ = Path(__file__).parent / 'resource'


def get_jina_latest_version() -> str:
    try:
        import json
        from urllib.request import Request, urlopen

        req = Request(
            'https://api.jina.ai/latest', headers={'User-Agent': 'Mozilla/5.0'}
        )

        with urlopen(req, timeout=1) as resource:
            latest_ver = json.load(resource)['version']

            return latest_ver
    except:
        return None
