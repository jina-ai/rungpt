from utils import generate_plain_prompts

import open_gpt
from open_gpt.profile import (
    compute_module_sizes,
    end_measure,
    log_measures,
    start_measure,
)

PROMPTS = generate_plain_prompts()

start_measures = start_measure()
model = open_gpt.create_model('RWKV/rwkv-raven-1b5', device='cpu', precision='fp16')
end_measures = end_measure(start_measures)
log_measures(end_measures, "Model loading")
module_sizes = compute_module_sizes(model)

start_measures = start_measure()
for prompt in PROMPTS:
    output = model.generate(prompt, max_length=100)
    print(output)

end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
