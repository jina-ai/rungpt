import run_gpt
from run_gpt.profile import end_measure, log_measures, start_measure

is_step = True

# start_measures = start_measure()
model = run_gpt.create_model(
    'decapoda-research/llama-7b-hf', precision='fp16', device_map='balanced'
)

prompt = 'The goal of life is'

if not is_step:
    generated_text = model.generate(
        prompt, max_new_tokens=10, do_sample=False, temperature=0.9
    )
    print(generated_text)
else:
    generated_text = model.step_generate(
        prompt, max_new_tokens=10, do_sample=False, temperature=0.9, stop_str=['happy']
    )
    for _ in generated_text:
        _['past_key_values'] = None
        print(_)
# end_measures = end_measure(start_measures)
# log_measures(end_measures, "Model generation")
