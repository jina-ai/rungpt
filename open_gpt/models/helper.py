from typing import List


def messages_to_prompt(messages: List[dict]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message['role']
        content = message['content']
        string_message = f"{role}: {content}"

        # addtional_kwargs = message.additional_kwargs
        # if addtional_kwargs:
        #     string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"assistant: ")
    return "\n".join(string_messages)