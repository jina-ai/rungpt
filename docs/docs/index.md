# Quick start

`opengpt` is an open-source _cloud-native_ large-scale **_multimodal models_** (LMMs) serving framework. 
It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs.
We aim to make it a one-stop solution for a centralized and accessible place to gather techniques for optimizing large-scale multimodal models and make them easy to use for everyone.


## Installation and setup

To use `opengpt`, install it with `pip`:

<div class="termy">

```shell
$ pip install open_gpt_torch
```

</div>


## Quick start

We use the [stabilityai/stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b) as the open example model as it is relatively small and fast to download.

<div class="termy">

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
    num_return_sequences=1,
)
```

</div>

On startup, the server sets up a default project that runs dev environments and tasks locally. 

!!! info "NOTE:"

    It usually takes a while (several minutes) for the first time to download and load the model into the memory.
