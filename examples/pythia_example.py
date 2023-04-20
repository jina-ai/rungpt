import time

from open_gpts.factory import create_model_and_transforms
from open_gpts.profile import (
    compute_module_sizes,
    end_measure,
    log_measures,
    start_measure,
)

start_measures = start_measure()
model, tokenizer, *_ = create_model_and_transforms(
    model_name='EleutherAI/pythia-12b-deduped'
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

module_sizes = compute_module_sizes(model)
device_size = {v: 0 for v in model.hf_device_map.values()}
for module, device in model.hf_device_map.items():
    device_size[device] += module_sizes[module]
message = "\n".join(
    [f"- {device}: {size // 2**20}MiB" for device, size in device_size.items()]
)
print(f"\nTheoretical use:\n{message}")

PROMPTS = [
    "Hello, my name is",
    "Are unicorns real? Unicorns are",
    "For the first time in several years,",
    "My name is Julien and I am",
    "The goal of life is",
    "Whenever I'm sad, I like to",
]

start_measures = start_measure()
generation_times = []
gen_tokens = []
texts_outs = []
for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = inputs["input_ids"][0].tolist()
    before_generate = time.time()
    outputs = model.generate(**inputs)
    after_generate = time.time()
    outputs = outputs[0].tolist()
    num_gen_tokens = (
        len(outputs) if outputs[: len(tokens)] != tokens else len(outputs) - len(tokens)
    )
    generation_time = after_generate - before_generate

    text_out = tokenizer.decode(outputs, skip_special_tokens=True)
    texts_outs.append(text_out)
    generation_times.append(generation_time)
    gen_tokens.append(num_gen_tokens)
    print(
        f"Prompt: {prompt}\nGeneration {text_out}\nIn {generation_time:.2f}s for {num_gen_tokens} tokens\n"
    )

end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")

generation_times_per_token = [
    gen / tok for gen, tok in zip(generation_times, gen_tokens)
]
avg_gen = sum(generation_times_per_token) / len(generation_times)
print(f"Average time of generation per token: {avg_gen:.2f}s")
print(f"First generation (avg time per token): {generation_times_per_token[0]:.2f}s")
avg_gen = sum(generation_times_per_token[1:]) / (len(generation_times_per_token) - 1)
print(f"Average time of generation per token (excluding the first): {avg_gen:.2f}s")
