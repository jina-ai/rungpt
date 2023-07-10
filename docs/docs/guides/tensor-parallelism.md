# Tensor-parallelism support

For transformer decoders used in large language models, tensor-parallelism is neccessary as it provides a way to 
shard the models' large weight matrices onto multiple NeuronCores, and having NeuronCores working on the same matrix multiply operation collaboratively. 
transformers-neuronx's tensor-parallelism support makes heavy use of collective operations such as all-reduce, which is supported natively by the Neuron runtime.

There are some principles for setting tensor-parallelism degree (number of NeuronCores participating in sharded matrix multiply operations) for Neuron-optimized transformer decoder models.

1. The number of attention heads needs to be divisible by the tensor-parallelism degree.
2. The total data size of model weights and key-value caches needs to be smaller than 16 GB times the tensor-parallelism degree.
3. Currently, the Neuron runtime supports tensor-parallelism degrees 1, 2, 8, and 32 on Trn1 and supports tensor-parallelism degrees 1, 2, 4, 8, and 24 on Inf2.

Some examples:

- `facebook/opt-13b` has 40 attention heads, and when running at batch size 1 and float16 precision the model requires ~29 GB memory, therefore a trn1.2xlarge with 32 GB device memory is sufficient.
- `facebook/opt-30b` has 56 attention heads, and at batch size 1 and float16 precision the model requires ~66 GB memory, therefore it can run on 8 NeuronCores on one trn1.32xlarge using 128 GB device memory.
- `gpt2-xl` has 25 attention heads and requires ~4 GB memory at bfloat16 precision. It runs without tensor-parallelism only.