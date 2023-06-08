from utils import generate_moss_prompts

import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

start_measures = start_measure()
model = open_gpt.create_model(
    model_name='fnlp/moss-moon-003-sft',
    precision='fp16',
    device_map='balanced',
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

prompt = generate_moss_prompts("你好", including_meta_instruction=True)

start_measures = start_measure()
generation_times = []
gen_tokens = []
texts_outs = []
text_out = model.generate(
    prompts=prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.02,
    max_new_tokens=256,
)
print(f"Prompt: {prompt}\nGeneration {text_out}")
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")

prompt = (
    prompt
    + text_out
    + "\n"
    + generate_moss_prompts("推荐五部科幻电影", including_meta_instruction=False)
)
start_measures = start_measure()
generation_times = []
gen_tokens = []
texts_outs = []
text_out = model.generate(
    prompts=prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.02,
    max_new_tokens=256,
)
print(f"Prompt: {prompt}\nGeneration {text_out}")
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
