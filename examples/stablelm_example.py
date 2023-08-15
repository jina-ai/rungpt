import run_gpt
from run_gpt.profile import end_measure, log_measures, start_measure

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

start_measures = start_measure()
model = run_gpt.create_model(
    model_name='stabilityai/stablelm-tuned-alpha-7b',
    precision='fp16',
    device_map='balanced',
)
# model = run_gpt.create_model(
#     model_name='stabilityai/stablelm-tuned-alpha-7b',
#     precision='bit4',
#     device_map='balanced',
# )

generated_text = model.generate(
    prompt, max_new_tokens=64, do_sample=True, temperature=0.7
)
print(f'==> {prompt} {generated_text}')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
