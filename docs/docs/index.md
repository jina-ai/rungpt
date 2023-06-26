# Introduction

`opengpt` is an open-source _cloud-native_ large-scale **_multimodal models_** (LMMs) serving framework. 
It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs.
We aim to make it a one-stop solution for a centralized and accessible place to gather techniques for optimizing large-scale multimodal models and make them easy to use for everyone.


## Installation

To use `opengpt`, install it with `pip`:

<div class="termy">

```shell
$ pip install open_gpt_torch
```

</div>

!!! info "NOTE:"

    To run **open_gpt** locally, it is required to have `Pytorch` pre-installed (see [Pytorch installation guide](https://pytorch.org/get-started/locally/)).


## Quick Start

We use the `stabilityai/stablelm-tuned-alpha-3b` model from the [huggingface](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b) as the open example as it is relatively small and fast to download.

<div editor-title="examples/quick_start.py">

```python
import open_gpt

model = open_gpt.create_model(
    'stabilityai/stablelm-tuned-alpha-3b', device='cuda', precision='fp16'
)

prompt = "The quick brown fox jumps over the lazy dog."

output = model.generate(
    prompt,
    max_length=100,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
)
```

</div>

In the above example, we use `precision='fp16'` to reduce the memory footprint and speed up the inference 
with some loss in accuracy on text generation tasks. You can also use `precision='fp32'` instead as you like for better performance.


!!! info "NOTE:"

    It usually takes a while (several minutes) for the first time to download and load the model into the memory.

## Serving Models

The `opengpt` package provides a simple and unified API for serving large models. 
You can use it to serve your own models without any extra effort, and start to serve your models with `serve` CLI simply by:

<div class="termy">

```shell
$ opengpt serve stabilityai/stablelm-tuned-alpha-3b \
                  --device cuda \
                  --precision fp16
```

</div>

Once the server is ready, you will see the following logs:

![](../../assets/images/server_ready.png){ width=800 }

As you can see, this will start a gRPC and HTTP server listening on `0.0.0.0:51001` and `0.0.0.0:51002` respectively by default. 

!!! info "NOTE:"

    You can inspect the available options with `opengpt serve --help`.


Then, you can access the model server with **gRPC** or **HTTP** API depending on your needs.

### gRPC Client API

To use the gRPC API, you can use the `opengpt.Client` module to connect to the model server:

<div editor-title="client.py">

```python
from open_gpt import Client

client = Client()

# connect to the model server
model = client.get_model(endpoint='grpc://0.0.0.0:51001')

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

</div>

### HTTP Client API

To use the HTTP API, you can simply send a `POST` request to the `/generate` endpoint:

<div editor-title="http_client.py">

```python
import requests

prompt = "The quick brown fox jumps over the lazy dog."

response = requests.post(
    "http://0.0.0.0:51002/generate",
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

</div>

