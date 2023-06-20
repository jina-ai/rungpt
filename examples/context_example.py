import open_gpt

model = open_gpt.create_model(
    'decapoda-research/llama-7b-hf', precision='fp16', device_map='balanced'
)

context = "The boy's name is Bob."
prompt = "What is the boy's name?"
max_new_tokens = 10


context_past_key_values = model.generate_context(context)

generated_text = model.step_generate(
    prompt,
    max_new_tokens=max_new_tokens,
    output_past_key_values=False,
    past_key_values=context_past_key_values,
)

print(f"===> context: {context}")
print(f"===> prompt: {prompt}")

for idx, item in enumerate(generated_text):
    if idx == max_new_tokens:
        print(f"===> output: {item['generated_text']}")
