import json

import requests
import sseclient

is_streaming_mode = True
prompt = 'Once upon a time,'

if is_streaming_mode:
    url = 'http://0.0.0.0:51002/generate_stream'
else:
    url = 'http://0.0.0.0:51002/generate'

response = requests.post(
    url,
    json={
        "prompt": prompt,
        "max_tokens": 15,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": False,
        "echo": True,
        "n": 1,
        "stop": '.',
    },
    stream=True if is_streaming_mode else False,
)

if is_streaming_mode:
    client = sseclient.SSEClient(response)
    for event in client.events():
        print(event.data)
else:
    print(response.json())
