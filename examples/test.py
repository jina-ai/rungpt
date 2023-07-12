import open_gpt

mode = 'streaming'

model = open_gpt.create_model(
    'decapoda-research/llama-7b-hf', precision='fp16', device_map='cuda:1'
)

prompt = 'Once upon a time,'
max_new_tokens = 10
past_key_values = None

parameters = {
    # "temperature": 0.9,
    # "top_k": 50,
    # "top_p": 0.95,
    # "repetition_penalty": 1.2,
    # "do_sample": False,
    # "num_return_sequences": 1,
}

if mode == 'streaming':
    for idx in range(max_new_tokens):
        generated_text = model.step_generate(
            prompt,
            remove_first_token=True if idx > 0 else False,
            past_key_values=past_key_values,
            max_new_tokens=1,
            **parameters
        )
        resp = next(generated_text)
        past_key_values = resp['past_key_values']
        prompt = resp['generated_text']

else:
    prompt = 'Once upon a time,'
    generated_text = model.step_generate(prompt, max_new_tokens=10, **parameters)
    for _, resp in enumerate(generated_text):
        print(resp['generated_text'])

# generated_text = model.generate(prompt, max_new_tokens=20, do_sample=False)
# print(generated_text)
