import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

start_measures = start_measure()
model = open_gpt.create_model(
    model_name='stabilityai/stablelm-tuned-alpha-7b',
    precision='fp16',
    device_map='balanced',
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")


system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


PROMPTS = [
    f"{system_prompt}<|USER|>Hello, my name is Toro.<|ASSISTANT|>",
    f"{system_prompt}<|USER|>Are unicorns real? Unicorns are",
    f"{system_prompt}<|ASSISTANT|>For the first time in several years,",
    f"{system_prompt}<|User|>My name is Julien and I am",
    f"{system_prompt}<|ASSISTANT|>The goal of life is",
    f"{system_prompt}<|ASSISTANT|>Whenever I'm sad, I like to",
]

start_measures = start_measure()
generation_times = []
gen_tokens = []
texts_outs = []
for prompt in PROMPTS:
    text_out = model.generate(
        prompts=prompt,
    )
    print(f"Prompt: {prompt}\nGeneration {text_out}")
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
