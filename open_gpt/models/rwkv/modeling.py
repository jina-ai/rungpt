from typing import List

from open_gpt.models.modeling import BaseModel
class RWKVModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, prompt: str,
                 max_length: int,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 repetition_penalty: float,
                 do_sample:bool,
                 num_return_sequences:int,
                 **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(inputs["input_ids"], max_new_tokens=max_length)
        output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        return output

    def encode(self, prompts: List[str], **kwargs):
        embedding = []
        def layer_hook(module, inp, out):
            embedding.append(out)

        inputs = self.tokenizer(prompts, return_tensors="pt",padding=True)
        hook = self.model.rwkv.embeddings.register_forward_hook(layer_hook)
        outputs = self.model.forward(inputs["input_ids"])
        hook.remove()
        return embedding