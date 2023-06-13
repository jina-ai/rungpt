import time

import requests
import torch
from PIL import Image

import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
    ).raw
)

start_measures = start_measure()
model = open_gpt.create_model(
    'openflamingo/OpenFlamingo-9B', precision='fp16', device='cuda', device_map=None
)


prompt = '<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of'

text_out = model.generate(
    prompt=prompt, inplace_images=[demo_image_one, demo_image_two, query_image]
)
print(f"Prompt: {prompt}\nGeneration {text_out}")
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
