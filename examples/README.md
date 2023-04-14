


## LLaMA

Before running the example, please make sure you have downloaded the pre-trained model and 
converted it to HF format. You can find the instructions in the [README](../README.md) file.

- convert the pre-trained model to HF format

```bash
python scripts/convert_llama_weights_to_hf.py --input_dir /path/to/llama/ --model_size 7B --output_dir llama_7B
```

- run the example 

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llama_example.py
```

> **Note:** the current implementation cannot work on multi-GPU yet.