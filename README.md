# OpenGPT

`OpenGPT` is an open-source _cloud-native_ large **multi-modal models** (LMMs) serving solution. 
It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs.

> The content of README.md is just a placeholder to remind me of what I want to do.

## Features

OpenGPT provides the following features to make it easy to deploy and serve large multi-modal models (LMMs) in production:

- Support for multi-modal models
- Scalable architecture for handling high traffic loads
- Optimized for low-latency inference
- Automatic model partitioning and distribution across multiple GPUs
- Centralized model management and monitoring
- REST API for easy integration with existing applications

You can learn more about OpenGPT’s [architecture in our documentation](https://opengpt.readthedocs.io/en/latest/).


## Roadmap

You can view our roadmap with features that are planned, started, and completed on the [Roadmap discussion](discussions/categories/roadmap) category.

## Installation

Install the package with pip:

```bash
pip install open_gpt
```

## Quickstart

```python
import open_gpts

model = open_gpts.create_model('facebook/llama-7b', device='cuda', precision='fp16')

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

We also provide some advanced features to allow you to host your models cost-effectively:

- **Offloading**: you can offload parts of the model to CPU to reduce the cost of inference.

- **Quantization**: you can quantize the model to reduce the cost of inference.

For more details, please see the [documentation](https://opengpt.readthedocs.io/en/latest/).

## Serving Models

You can serve your models with OpenGPT. To do so, you can use the `serve` command:

```bash
opengpt serve facebook/llama-9b --device cuda --precision fp16 --port 5000
```

This will start a server on port 5000. You can then send requests to the server:

```python
import requests

prompt = "The quick brown fox jumps over the lazy dog."

response = requests.post(
    "http://localhost:5000/generate",
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


# SSE support
from aiohttp_sse_client import client as sse_client

async with sse_client.EventSource(
    'http://localhost:5000/stream/generate?prompt=The+quick+brown+fox+jumps+over+the+lazy+dog.&max_length=100&temperature=0.9&top_k=50&top_p=0.95&repetition_penalty=1.2&do_sample=True&num_return_sequences=1'
) as event_source:
    try:
        async for event in event_source:
            print(event)
    except ConnectionError:
        pass
```

Note that the server will only accept requests from the same machine. If you want to accept requests from other machines, you can use the `--host` flag to specify the host to bind to.

## Deploying Models

You can also deploy the server to a cloud provider like Jina Cloud or AWS.
To do so, you can use `deploy` command:

- Jina Cloud

```bash
opengpt deploy facebook/llama-9b --device cuda --precision fp16 --provider jina --name opengpt --replicas 2
```

- AWS

To deploy to AWS, you need to install extra dependencies: 

```bash
pip install opegpt[aws]
```

And you need to specify the region:

```bash
opengpt deploy facebook/llama-9b --device cuda --precision fp16 --provider aws --region us-east-1 --name opengpt --replicas 2
```

This will deploy the model to the cloud provider. You can then send requests to the server:

```python
import requests

prompt = "The quick brown fox jumps over the lazy dog."

response = requests.post(
    "https://opengpt.jina.ai/generate",
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

## Kubernetes

To deploy OpenGPT on your Kubernetes cluster, follow these steps:

1. Install the OpenGPT operator on your Kubernetes cluster using Helm:

    ```bash
    helm install opengpt ./helm/opengpt --namespace opengpt
    ```

2. Create a custom resource for your GPT model:
    
    ```YAML
    apiVersion: opengpt.io/v1alpha1
    kind: GptModel
    metadata:
      name: my-gpt-model
      namespace: opengpt
    spec:
      modelPath: s3://my-bucket/my-model
      modelName: my-model
      maxBatchSize: 16
      inputShape:
        - 1024
        - 1024
        - 3
      outputShape:
        - 1024
        - 1024
        - 3

    ```
   
3. Apply the custom resource to your cluster:

    ```bash
   kubectl apply -f my-gpt-model.yaml
    ```

4. Monitor the status of your GPT model using the OpenGPT dashboard:

    ```bash
   kubectl port-forward -n opengpt svc/opengpt-dashboard 8080:80
    ```

## Accessing models via API

You can also access the online models via API. To do so, you can use the `inference_client` package:

```python
from inference_client import Client

client = Client(token='<your access token>')

model = client.get_model('facebook/llama-9b')

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

By this way, you can access the models without deploying them to your own machine.

## Advanced Usage

### Model Offloading

You can also apply the model offloading techniques (based on [FlexTensor](https://github.com/numb3r3/flex-tensor)) to OpenGPT. To do so, you can use the `--offload-percents` flag:

```bash
opengpt serve facebook/llama-9b --device cuda --precision fp16 --port 5000 --offload-percents 10,90,50,50,0,100
```

This will offload parts of the model to the CPU. You can also use the `--offload-strategy` flag to specify the offloading strategy:

```bash
opengpt serve facebook/llama-9b --device cuda --precision fp16 --port 5000 --offload-strategy "cpu,cpu,cpu,cpu,cpu,cpu"
```

### Model Quantization

You can also apply the model quantization techniques.

- 8-bit quantization

```bash
opengpt serve facebook/llama-9b --device cuda --precision fp16 --port 5000 --quantize 8bit
```

## Fine-tuning Models

We currently support fine-tuning models by using the `finetune` command:

```bash
opengpt finetune facebook/llama-9b --dataset wikitext-2 --device cuda --precision fp16 --batch-size 32 --learning-rate 1e-4 --epochs 10
```

Specifically, we implement the following fine-tuning methods:
- [LLaMA-Adapter: Efficient Fine-tuning of LLaMA](https://github.com/ZrrSkywalker/LLaMA-Adapter): Fine-tuning model to follow instructions within 1 Hour and 1.2M Parameters
- [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf): Low-Rank Adaptation for Efficient Language Model Fine-Tuning

## Documentation

For more information, check out the [documentation](https://opengpt.readthedocs.io/en/latest/).


## Contributing

We welcome contributions from the community! To contribute, please submit a pull request following our contributing guidelines.

## License

OpenGPT is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.