# ☄️ RunGPT

<p align="center">
<a href="https://github.com/jina-ai/rungpt"><img src="https://github.com/jina-ai/rungpt/blob/main/.github/images/logo.png" alt="RunGPT: An open-source cloud-native large-scale multimodal model serving framework" width="300px"></a>
<br>
</p>

> "A playful and whimsical vector art of a Stochastic Tigger, wearing a t-shirt with a "GPT" text printed logo, surrounded by colorful geometric shapes.  –ar 1:1 –upbeta"
>
> — Prompts and logo art was produced with  [PromptPerfect](https://promptperfect.jina.ai/) & [Stable Diffusion X](https://clipdrop.co/stable-diffusion)


![](https://img.shields.io/badge/Made%20with-JinaAI-blueviolet?style=flat)
[![PyPI](https://img.shields.io/pypi/v/run_gpt_torch)](https://pypi.org/project/run_gpt_torch/)
[![PyPI - License](https://img.shields.io/pypi/l/run_gpt_torch)](https://pypi.org/project/run_gpt_torch/)

**RunGPT** is an open-source _cloud-native_ **_large-scale language models_** (LLMs) serving framework. 
It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs.
We aim to make it a one-stop solution for a centralized and accessible place to gather techniques for optimizing LLM and make them easy to use for everyone.


## Table of contents

- [Features](#features)
- [Get started](#get-started)
- [Build a model serving in one line](#build-a-model-serving-in-one-line)
- [Cloud-native deployment](#cloud-native-deployment)
- [Roadmap](#roadmap)

## Features

RunGPT provides the following features to make it easy to deploy and serve **large language models** (LLMs) at scale:

- Scalable architecture for handling high traffic loads
- Optimized for low-latency inference
- Automatic model partitioning and distribution across multiple GPUs
- Centralized model management and monitoring
- REST API for easy integration with existing applications

## Updates

- **2023-08-22**: The OpenGPT is now renamed to RunGPT. We have also released the first version `v0.1.0` of RunGPT. You can install it with `pip install rungpt`.
- **2023-05-12**: 🎉We have released the first version `v0.0.1` of OpenGPT. You can install it with `pip install open_gpt_torch`.


## Get Started

### Installation

Install the package with `pip`:

```bash
pip install rungpt
```

### Quickstart

```python
import run_gpt

model = run_gpt.create_model(
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
    num_return_sequences=1,
)
```

We use the [stabilityai/stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b) as the open example model as it is relatively small and fast to download.

> **Warning**
> In the above example, we use `precision='fp16'` to reduce the memory usage and speed up the inference with some loss in accuracy on text generation tasks. 
> You can also use `precision='fp32'` instead as you like for better performance. 

> **Note**
> It usually takes a while (several minutes) for the first time to download and load the model into the memory.


In most cases of large model serving, the model cannot fit into a single GPU. To solve this problem, we also provide a `device_map` option (supported by `accecleate` package) to automatically partition the model and distribute it across multiple GPUs:

```python
model = run_gpt.create_model(
    'stabilityai/stablelm-tuned-alpha-3b', precision='fp16', device_map='balanced'
)
```

In the above example, `device_map="balanced"` evenly split the model on all available GPUs, making it possible for you to serve large models.

> **Note**
> The `device_map` option is supported by the [accelerate](https://github.com/huggingface/accelerate) package. 


See [examples on how to use rungpt with different models.](./examples) 🔥


## Build a model serving in one line

To do so, you can use the `serve` command:

```bash
rungpt serve stabilityai/stablelm-tuned-alpha-3b --precision fp16 --device_map balanced
```

💡 **Tip**: you can inspect the available options with `rungpt serve --help`.

This will start a gRPC and HTTP server listening on port `51000` and `52000` respectively. 
Once the server is ready, as shown below:
<details>
<summary>Click to expand</summary>
<img src="https://github.com/jina-ai/rungpt/blob/main/.github/images/serve_ready.png" width="600px">
</details>

You can then send requests to the server:

```python
import requests

prompt = "Once upon a time,"

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

What's more, we also provide a [Python client](https://github.com/jina-ai/inference-client/) (`inference-client`) for you to easily interact with the server:

```python
from run_gpt import Client

client = Client()

# connect to the model server
model = client.get_model(endpoint='grpc://0.0.0.0:51000')

prompt = "Once upon a time,"

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

The output has the same format as the one from the OpenAI's Python API:

```
{ "id": "18d92585-7b66-4b7c-b818-71287c122c50", 
  "object": "text_completion", 
  "created": 1692610173, 
  "choices": [{"text": "Once upon a time, there was an old man who lived in the forest. He had no children", 
              "finish_reason": "length", 
              "index": 0.0}], 
  "prompt": "Once upon a time,", 
  "usage": {"completion_tokens": 21, "total_tokens": 27, "prompt_tokens": 6}}
```

For the streaming output, you can install `sseclient-py` first:
```bash
pip install sseclient-py
```

And send the request to `http://localhost:51000/generate_stream` with the same payload.

```python
import sseclient
import requests

prompt = "Once upon a time,"

response = requests.post(
    "http://localhost:51000/generate_stream",
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
    stream=True,
)
client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)
```

And the output will be streamed back to you (only show 3 iterations here):

```
{ "id": "18d92585-7b66-4b7c-b818-71287c122c51", 
  "object": "text_completion", 
  "created": 1692610173, 
  "choices": [{"text": " there", "finish_reason": None, "index": 0.0}], 
  "prompt": "Once upon a time,", 
  "usage": {"completion_tokens": 1, "total_tokens": 7, "prompt_tokens": 6}},
{ "id": "18d92585-7b66-4b7c-b818-71287c122c52", 
  "object": "text_completion", 
  "created": 1692610173, 
  "choices": [{"text": "was", "finish_reason": None, "index": 0.0}], 
  "prompt": None, 
  "usage": {"completion_tokens": 2, "total_tokens": 9, "prompt_tokens": 7}},
{ "id": "18d92585-7b66-4b7c-b818-71287c122c53", 
  "object": "text_completion", 
  "created": 1692610173, 
  "choices": [{"text": "an", "finish_reason": None, "index": 0.0}], 
  "prompt": None, 
  "usage": {"completion_tokens": 3, "total_tokens": 11, "prompt_tokens": 8}}
```

We also support chat mode, which is useful for interactive applications. The inputs for `chat` should be a list of 
dictionaries which contain role and content. For example:

```python
import requests

messages = [
    {"role": "user", "content": "Hello!"},
]

response = requests.post(
    "http://localhost:51000/chat",
    json={
        "messages": messages,
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

The response will be:

```
{"id": "18d92585-7b66-4b7c-b818-71287c122c57", 
  "object": "chat.completion", 
  "created": 1692610173, 
  "choices": [{"message": {
                            "role": "assistant",
                            "content": "\n\nHello there, how may I assist you today?",
                        }, 
              "finish_reason": "stop", "index": 0.0}], 
  "prompt": "Hello there!", 
  "usage": {"completion_tokens": 12, "total_tokens": 15, "prompt_tokens": 3}}
```

You can also replace the `chat` with `chat_stream` to get the streaming output.



## Cloud-native deployment

You can also deploy the server to a cloud provider like Jina Cloud or AWS.
To do so, you can use `deploy` command:

### Jina Cloud

using predefined executor

```bash
rungpt deploy stabilityai/stablelm-tuned-alpha-3b --precision fp16 --device_map balanced --cloud jina --replicas 1
```

It will give you a HTTP url and a gRPC url by default:
```bash
https://{random-host-name}-http.wolf.jina.ai
grpcs://{random-host-name}-grpc.wolf.jina.ai
```


### AWS

TBD


## Benchmark
We have done some benchmarking on different model architectures and different configurations (whether to use 
quantization, torch.compile and page attention ...), regards to the latency, throughput (prefill stage && the whole 
decoding process) and perplexity. 

The script for benchmarking locates at `scripts/benchmark.py`. You can run the scripts to get the benchmarking results.

### Environment Setting
We use a single RTX3090 (cuda version is 11.8) for all benchmarking except for Llama-2-13b (2*RTX3090). We use:
```
torch==2.0.1 (without torch.compile) / torch==2.1.0.dev20230803 (with torch.compile)
bitsandbytes==0.41.0
transformers==4.31.0
triton==2.0.0
```

### Model Candidates
|             Model_Name             |
|:----------------------------------:|
|      meta-llama/Llama-2-7b-hf      |
|           mosaicml/mpt-7b          |
| stabilityai/stablelm-base-alpha-7b |
|         EleutherAI/gpt-j-6B        |


### Benchmarking Results

- **Latency/throughput for different models** (precision: fp16)

|             Model_Name             | average_prefill_latency(ms/token) | average_prefill_throughput(token/s) | average_decode_latency(ms/token) | average_decode_throughput(token/s) |
|:----------------------------------:|:---------------------------------:|:-----------------------------------:|:--------------------------------:|:----------------------------------:|
|      meta-llama/Llama-2-7b-hf      |                 49                |                20.619               |               49.4               |               20.054               |
|      meta-llama/Llama-2-13b-hf     |                175                |                5.727                |              188.27              |                4.836               |
|           mosaicml/mpt-7b          |                 27                |                37.527               |               28.04              |               35.312               |
| stabilityai/stablelm-base-alpha-7b |                 50                |                20.09                |               45.73              |               21.878               |
|         EleutherAI/gpt-j-6B        |                 75                |                13.301               |               76.15              |               11.181               |


- **Latency/throughput for different models using torch.compile** (precision: fp16)

> **Warning**
> torch.compile doesn't support Flash-Attention based model like MPT. Also, it cannot be used in multi-GPUs environment.

|             Model_Name             | average_prefill_latency(ms/token) | average_prefill_throughput(token/s) | average_decode_latency(ms/token) | average_decode_throughput(token/s) |
|:----------------------------------:|:---------------------------------:|:-----------------------------------:|:--------------------------------:|:----------------------------------:|
|      meta-llama/Llama-2-7b-hf      |                 25                |                40.644               |               26.54              |                37.75               |
|      meta-llama/Llama-2-13b-hf     |                 -                 |                  -                  |                 -                |                  -                 |
|           mosaicml/mpt-7b          |                 -                 |                  -                  |                 -                |                  -                 |
| stabilityai/stablelm-base-alpha-7b |                 44                |                22.522               |               42.97              |               21.413               |
|         EleutherAI/gpt-j-6B        |                 32                |                31.488               |               33.89              |               25.105               |

- **Latency/throughput for different models using quantization** (precision: fp16 / bit8 / bit4)

<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">prefill latency (ms/token)</th>
    <th colspan="3">prefill throughput (tokens/s)</th>
    <th colspan="3">decode latency (ms/token)</th>
    <th colspan="3">decode throughput (tokens/s)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-7b-hf</td>
    <td>49</td>
    <td>301</td>
    <td>125</td>
    <td>20.619</td>
    <td>3.325</td>
    <td>8.015</td>
    <td>49.4</td>
    <td>256.44</td>
    <td>112.22</td>
    <td>20.054</td>
    <td>3.9</td>
    <td>8.918</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-13b-hf</td>
    <td>175</td>
    <td>974</td>
    <td>376</td>
    <td>5.727</td>
    <td>1.027</td>
    <td>2.662</td>
    <td>182.27</td>
    <td>796.32</td>
    <td>349.93</td>
    <td>4.836</td>
    <td>1.144</td>
    <td>2.662</td>
  </tr>
  <tr>
    <td>mosaicml/mpt-7b</td>
    <td>27</td>
    <td>139</td>
    <td>86</td>
    <td>37.527</td>
    <td>7.222</td>
    <td>11.6</td>
    <td>28.04</td>
    <td>141.04</td>
    <td>94.22</td>
    <td>35.312</td>
    <td>7.021</td>
    <td>10.507</td>
  </tr>
  <tr>
    <td>stabilityai/stablelm-base-alpha-7b</td>
    <td>50</td>
    <td>164</td>
    <td>156</td>
    <td>20.09</td>
    <td>6.134</td>
    <td>6.408</td>
    <td>45.73</td>
    <td>148.53</td>
    <td>147.56</td>
    <td>21.878</td>
    <td>6.947</td>
    <td>6.994</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6B</td>
    <td>75</td>
    <td>368</td>
    <td>162</td>
    <td>13.301</td>
    <td>2.724</td>
    <td>6.195</td>
    <td>76.15</td>
    <td>365.51</td>
    <td>138.44</td>
    <td>11.181</td>
    <td>2.327</td>
    <td>5.642</td>
  </tr>
</tbody>
</table>


- **Perplexity for different models using quantization** (precision: fp16 / bit8 / bit4)

> **Notice**
> From this benchmark we see that quantization doesn't affect the perplexity of the model too much.

<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">wikitext2</th>
    <th colspan="3">ptb </th>
    <th colspan="3">c4</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
    <td>fp16</td>
    <td>bit8</td>
    <td>bit4</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-7b-hf</td>
    <td>5.4721</td>
    <td>5.506</td>
    <td>5.6437</td>
    <td>22.9483</td>
    <td>23.8797</td>
    <td>25.0556</td>
    <td>6.9727</td>
    <td>7.0098</td>
    <td>7.1623</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-13b-hf</td>
    <td>4.8837</td>
    <td>4.9229</td>
    <td>4.9811</td>
    <td>27.6802</td>
    <td>27.9665</td>
    <td>28.8417</td>
    <td>6.4677</td>
    <td>6.4884</td>
    <td>6.566</td>
  </tr>
  <tr>
    <td>mosaicml/mpt-7b</td>
    <td>7.6829</td>
    <td>7.7256</td>
    <td>7.9869</td>
    <td>10.6002</td>
    <td>10.6743</td>
    <td>10.9486</td>
    <td>9.6001</td>
    <td>9.6457</td>
    <td>9.879</td>
  </tr>
  <tr>
    <td>stabilityai/stablelm-base-alpha-7b</td>
    <td>14.1886</td>
    <td>14.268</td>
    <td>15.9817</td>
    <td>19.2968</td>
    <td>19.4904</td>
    <td>21.3513</td>
    <td>48.222</td>
    <td>48.3384</td>
    <td>57.022</td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6B</td>
    <td>8.8563</td>
    <td>8.8786</td>
    <td>9.0301</td>
    <td>13.5946</td>
    <td>13.6137</td>
    <td>13.784</td>
    <td>11.7114</td>
    <td>11.7293</td>
    <td>11.8929</td>
  </tr>
</tbody>
</table>

- **Latency/throughput for different models using vllm** (precision: fp16)

> **Warning**
> vllm brings a significant improvement in latency and throughput, but it is not compatible with streaming output, so we don't release it yet.

<table>
<thead>
  <tr>
    <th></th>
    <th colspan="2">prefill latency (ms/token)</th>
    <th colspan="2">prefill throughput (tokens/s)</th>
    <th colspan="2">decode latency (ms/token)</th>
    <th colspan="2">decode throughput (tokens/s)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>using vllm</td>
    <td>baseline</td>
    <td>using vllm</td>
    <td>baseline</td>
    <td>using vllm</td>
    <td>baseline</td>
    <td>using vllm</td>
    <td>baseline</td>
  </tr>
  <tr>
    <td>meta-llama/Llama-2-7b-hf</td>
    <td>29</td>
    <td>49</td>
    <td>34.939</td>
    <td>20.619</td>
    <td>20.34</td>
    <td>49.40</td>
    <td>48.67</td>
    <td>20.054</td>
  </tr>
</tbody>
</table>

## Contributing

We welcome contributions from the community! To contribute, please submit a pull request following our contributing guidelines.

## License

RunGPT is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.