import run_gpt
from run_gpt.profile import end_measure, log_measures, start_measure

start_measures = start_measure()
model = run_gpt.create_model(
    'Salesforce/codegen-350M-mono', precision='fp16', device_map='balanced'
)

prompt = 'def hello_world():'


generated_text = model.generate(
    prompt, max_new_tokens=256, do_sample=False, temperature=0.9
)
print(f'==> {prompt} {generated_text}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
