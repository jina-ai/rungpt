import run_gpt
from run_gpt.profile import end_measure, log_measures, start_measure

is_step = False

start_measures = start_measure()
model = run_gpt.create_model(
    'nthngdy/pythia-owt2-70m-100k', precision='fp16', device_map='balanced'
)

message = [
    {"role": "user", "content": "What is your name?"},
    {"role": "assistant", "content": "My name is Pythia."},
    {
        "role": "user",
        "content": 'Hello Pythia, can you tell me that is the goal of life?',
    },
]

if not is_step:
    generated_text = model.chat(
        message, max_new_tokens=10, do_sample=False, temperature=0.9, stop_str=['\n']
    )
    print(generated_text)
else:
    generated_text = model.step_chat(
        message, max_new_tokens=10, do_sample=False, temperature=0.9, stop_str=['\n']
    )
    for _ in generated_text:
        _['past_key_values'] = None
        print(_)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
