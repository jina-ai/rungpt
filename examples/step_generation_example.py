import json

import requests
import sseclient

url = 'http://0.0.0.0:51002/generate_stream'
response = requests.post(
    url,
    json={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "max_new_tokens": 10,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": False,
        "num_return_sequences": 1,
    },
    stream=True,
)
client = sseclient.SSEClient(response)
for event in client.events():
    print(json.loads(event.data))
