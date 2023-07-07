import requests
from sseclient import SSEClient


def handle_event(event):
    # Extract the event data
    event_id = event.id
    event_data = event.data
    event_type = event.event

    # Process the event data
    # Replace this with your own logic
    print(f"Event ID: {event_id}")
    print(f"Event Type: {event_type}")
    print(f"Event Data: {event_data}")


def consume_sse_events(url, prompt):
    response = requests.post(
        url=url,
        json={
            "prompt": prompt,
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
    client = SSEClient(response)

    for event in client.events():
        handle_event(event)


prompt = "The quick brown fox jumps over the lazy dog."

sse_url = "http://0.0.0.0:51002/generate_stream"
consume_sse_events(sse_url, prompt)
