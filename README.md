# ☄️ OpenGPT

*An open-source cloud-native large-scale multimodal model serving framework*

![](https://img.shields.io/badge/Made%20with-JinaAI-blueviolet?style=flat)
[![PyPI](https://img.shields.io/pypi/v/open_gpt_torch)](https://pypi.org/project/open_gpt_torch/)
[![PyPI - License](https://img.shields.io/pypi/l/open_gpt_torch)](https://pypi.org/project/open_gpt_torch/)

`OpenGPT` is an open-source _cloud-native_ large-scale **multimodal models** (LMMs) serving framework. 
It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs.

## Table of contents

- [Features](#features)
- [Supported models](#supported-models)
- [Get started](#get-started)
- [Build a model serving in one line](#build-a-model-serving-in-one-line)
- [Cloud-native deployment](#cloud-native-deployment)
- [Roadmap](#roadmap)

## Features

OpenGPT provides the following features to make it easy to deploy and serve **large multi-modal models** (LMMs) at scale:

- Support for multi-modal models on top of large language models
- Scalable architecture for handling high traffic loads
- Optimized for low-latency inference
- Automatic model partitioning and distribution across multiple GPUs
- Centralized model management and monitoring
- REST API for easy integration with existing applications

## Updates

- **2023-05-12**: 🎉 We have released the first version `v0.0.1` of OpenGPT. You can install it with `pip install open_gpt_torch`.

## Supported Models

OpenGPT supports the following models out of the box:

- LLM (Large Language Model)

  - [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/): open and efficient foundation language models by Meta
  - [Pythia](https://github.com/EleutherAI/pythia): a collection of models developed to facilitate interpretability research by EleutherAI
  - [StableLM](https://github.com/Stability-AI/StableLM): series of large language models by Stability AI
  - [Vicuna](https://vicuna.lmsys.org/): a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS
  - [MOSS](https://txsun1997.github.io/blogs/moss.html): conversational language model from Fudan University

- LMM (Large Multi-modal Model)

  - [OpenFlamingo](https://github.com/mlfoundations/open_flamingo): an open source version of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model
  - [MiniGPT-4](https://minigpt-4.github.io/): aligns a frozen visual encoder with a frozen LLM, Vicuna, using just one projection layer. 

For more details about the supported models, please see the [Model Zoo](./MODEL_ZOO.md).


## Roadmap

You can view our roadmap with features that are planned, started, and completed on the [Roadmap discussion](https://github.com/jina-ai/opengpt/discussions/categories/roadmap) category.

## Get Started

### Installation

Install the package with `pip`:

```bash
pip install open_gpt_torch
```

### Quickstart

```python
import open_gpt

model = open_gpt.create_model('facebook/llama-7b', device='cuda', precision='fp16')

prompt = "The quick brown fox jumps over the lazy dog."

output = model.generate(
    prompt,
    max_length=100,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1,
)
```

We also provide some advanced features to allow you to host your models by using multiple GPUs:
    
```python
model = open_gpt.create_model(
    'facebook/llama-7b', device='cuda', precision='fp16', device_map='balanced'
)
```


## Build a model serving in one line

To do so, you can use the `serve` command:

```bash
opengpt serve facebook/llama-7b --precision fp16 --port 51000
```

This will start a server on port `51000`. You can then send requests to the server:

```python
import requests

prompt = "The quick brown fox jumps over the lazy dog."

response = requests.post(
    "http://localhost:51000/generate",
    json={
        "prompt": prompt,
        "max_length": 100,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "num_return_sequences": 1,
    },
)
```

Note that the server will only accept requests from the same machine. If you want to accept requests from other machines, you can use the `--host` flag to specify the host to bind to.

## Cloud-native deployment

You can also deploy the server to a cloud provider like Jina Cloud or AWS.
To do so, you can use `deploy` command:

- Jina Cloud

```bash
opengpt deploy facebook/llama-7b --device cuda --precision fp16 --provider jina --name opengpt --replicas 2
```

**TBD ...**


## Contributing

We welcome contributions from the community! To contribute, please submit a pull request following our contributing guidelines.

## License

OpenGPT is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.