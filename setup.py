import sys
from os import path

from setuptools import find_packages, setup

if sys.version_info < (3, 8, 0):
    raise OSError(f'OpenGPT requires Python >=3.8, but yours is {sys.version}')

try:
    pkg_name = 'open-gpt-torch'
    libinfo_py = path.join(path.dirname(__file__), 'open_gpt', '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = '0.0.0'


try:
    with open('../README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='An open-source cloud-native of large multi-modal models (LMMs) serving framework.',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='Apache 2.0',
    url='https://https://github.com/jina-ai/opengpt',
    download_url='https://https://github.com/jina-ai/opengpt/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=[
        'pydantic<2.0.0',
        'jina>=3.17.0',
        'docarray<0.30.0',
        'jcloud>0.2.0',
        'inference-client>=0.0.4',
        'cleo>=2.0.0',
        'click>=8.0.0',
        'Jinja2>=3.1.0',
        'sentencepiece>=0.1.96',
        'transformers>=4.30.1',
        'bitsandbytes>=0.39.0',
        'accelerate>=0.20.3',
        'peft>=0.3.0',
        'loguru',
        'tqdm',
        'sse_starlette>=1.6.1',
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pytest-xdist', 'pytest-mock'],
    },
    entry_points={
        'console_scripts': [
            'opengpt = open_gpt.cli.application:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        'Documentation': 'https://opengpt.jina.ai',
        'Source': 'https://github.com/jina-ai/opengpt/',
        'Tracker': 'https://github.com/jina-ai/opengpt/issues',
    },
    keywords=[
        "jina",
        "pytorch",
        "large-language-model",
        "GPT",
        "LLM",
        "multi-modality",
        "cloud-native",
        "model-serving",
        "model-inference",
        "llama",
        "vicuna",
        "stabellm",
    ],
)
