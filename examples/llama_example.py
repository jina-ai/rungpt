import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

start_measures = start_measure()
# model = open_gpt.create_model(
#     'decapoda-research/llama-7b-hf', precision='fp16', device_map='balanced'
# )

model = open_gpt.create_model(
    'yahma/llama-7b-hf', precision='bit8', device_map='balanced'
)

# model = open_gpt.create_model(
#     'openlm-research/open_llama_7b_700bt_preview',
#     precision='bit8',
#     device_map='balanced',
# )

prompt = 'The goal of life is'

end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

start_measures = start_measure()
generated_text = model.generate(
    prompt, max_new_tokens=256, do_sample=True, temperature=0.9
)
print(f'==> {prompt} {generated_text}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
