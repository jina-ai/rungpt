from utils import generate_stablelm_prompts

import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

prompt = generate_stablelm_prompts(
    "What's your mood today?", including_meta_instruction=True
)

start_measures = start_measure()
# model = open_gpt.create_model(
#     model_name='stabilityai/stablelm-tuned-alpha-7b',
#     precision='fp16',
#     device_map='balanced',
# )
model = open_gpt.create_model(
    model_name='stabilityai/stablelm-tuned-alpha-7b',
    precision='bit4',
    device_map='balanced',
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

start_measures = start_measure()
generation_times = []
gen_tokens = []
texts_outs = []
text_out = model.generate(
    prompts=prompt, max_new_tokens=256, do_sample=True, temperature=0.9
)
print(f"Prompt: {prompt}\nGeneration {text_out}")
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
