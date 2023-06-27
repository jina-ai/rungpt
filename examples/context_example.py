import open_gpt
from open_gpt.models.session import SessionManager

session_manager = SessionManager()

model = open_gpt.create_model(
    'decapoda-research/llama-7b-hf', precision='fp16', device_map='balanced'
)

context = "The boy's name is Bob."
prompt = "What is the boy's name?"
max_new_tokens = 10
session_id = 'this-is-a-fake-session-id'

generated_text = model.step_generate(
    context,
    max_new_tokens=max_new_tokens,
)

for idx, item in enumerate(generated_text):
    if idx == max_new_tokens:
        past_key_values = item['past_key_values']

session_manager.update(session_id, past_key_values)

generated_text = model.step_generate(
    prompt,
    max_new_tokens=max_new_tokens,
    past_key_values=session_manager.get(session_id),
)

for idx, item in enumerate(generated_text):
    if idx == max_new_tokens:
        past_key_values = item['past_key_values']
