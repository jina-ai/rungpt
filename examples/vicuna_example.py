from utils import generate_plain_prompts

import open_gpt
from open_gpt.profile import (
    compute_module_sizes,
    end_measure,
    log_measures,
    start_measure,
)

PROMPTS = generate_plain_prompts()


start_measures = start_measure()
model = open_gpt.create_model(
    'lmsys/vicuna-7b-delta-v1.1', precision='fp16', device_map='balanced'
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

start_measures = start_measure()
for prompt in PROMPTS:
    generated_text = model.generate(
        prompts=prompt, max_new_tokens=50, do_sample=True, temperature=0.9
    )
    print(f'==> {prompt} {generated_text}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation (single)")


# batch style
start_measures = start_measure()
generated_text = model.generate(
    prompts=PROMPTS, max_new_tokens=50, do_sample=True, temperature=0.9
)
for s, p in zip(generated_text, PROMPTS):
    print(f'==> {p} {s}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation (batch)")
