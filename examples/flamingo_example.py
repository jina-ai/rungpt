import time

import requests
import torch
from PIL import Image

from open_gpt.factory import create_model_and_transforms
from open_gpt.profile import (
    compute_module_sizes,
    end_measure,
    log_measures,
    start_measure,
)

start_measures = start_measure()
model, tokenizer, image_processor = create_model_and_transforms(
    model_name='openflamingo/OpenFlamingo-9B'
)
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")

# module_sizes = compute_module_sizes(model)
# device_size = {v: 0 for v in model.hf_device_map.values()}
# for module, device in model.hf_device_map.items():
#     device_size[device] += module_sizes[module]
# message = "\n".join([f"- {device}: {size // 2**20}MiB" for device, size in device_size.items()])
# print(f"\nTheoretical use:\n{message}")


"""
Step 1: Load images
"""
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
        "https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png",
        stream=True,
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
vision_x = [
    image_processor(demo_image_one).unsqueeze(0),
    image_processor(demo_image_two).unsqueeze(0),
    image_processor(query_image).unsqueeze(0),
]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
lang_x = tokenizer(
    [
        "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"
    ],
    return_tensors="pt",
)

"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_inputs=vision_x,
    text_inputs=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=6,
    top_p=0.9,
    no_repeat_ngram_size=2,
    temperature=0.7,
    length_penalty=1.5,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
