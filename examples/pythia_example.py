import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

start_measures = start_measure()
model = open_gpt.create_model(
    'EleutherAI/pythia-12b-deduped', precision='fp16', device_map='balanced'
)

# model = open_gpt.create_model(
#     'EleutherAI/pythia-12b-deduped', precision='bit4', device_map='balanced'
# )

prompt = 'The goal of life is'


generated_text = model.generate(
    prompt, max_new_tokens=256, do_sample=True, temperature=0.9
)
print(f'==> {prompt} {generated_text}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
