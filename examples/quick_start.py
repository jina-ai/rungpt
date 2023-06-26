import open_gpt

model = open_gpt.create_model(
    'stabilityai/stablelm-tuned-alpha-3b', device='cuda', precision='fp16'
)

prompt = "The quick brown fox jumps over the lazy dog."

output = model.generate(
    prompt,
    max_length=100,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
)
