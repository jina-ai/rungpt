from enum import Enum
from typing import Any, List, Tuple, Union

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    # session id
    id: str = Field(description='The session id of the generation.', default=None)

    # generation parameters
    num_beams: int = Field(description='The number of beams to use.', default=None)
    max_tokens: int = Field(
        description='The maximum length of the generated text.', default=None
    )
    max_new_tokens: int = Field(
        description='The maximum length of the generated text.', default=None
    )
    temperature: float = Field(
        description='The temperature of the generation.', default=None
    )
    top_k: int = Field(description='The top k of the generation.', default=None)
    top_p: float = Field(description='The top p of the generation.', default=None)
    repetition_penalty: float = Field(
        description='The repetition penalty of the generation.', default=None
    )
    logprobs: int = Field(
        description='Include the log probabilities on the logprobs '
        'most likely tokens, as well the chosen tokens',
        default=None,
    )
    echo: bool = Field(
        description='Echo back the prompt in the completion.', default=None
    )
    stop: Union[str, List[str]] = Field(
        description='Stop sequence generation on token.', default=None
    )
    stop_str: Union[str, List[str]] = Field(
        description='Stop sequence generation on token.', default=None
    )
    do_sample: bool = Field(
        description='Whether to sample from the generation.', default=None
    )
    presence_penalty: float = Field(
        description='Positive values penalize new tokens based on whether they appear in '
        'the text so far, increasing the likelihood to talk about new topics.',
        default=0,
    )
    frequency_penalty: float = Field(
        description='Positive values penalize new tokens based on their existing '
        'frequency in the text so far, decreasing the likelihood to repeat '
        'the same line verbatim.',
        default=0,
    )
    best_of: int = Field(
        description='Generates best_of completions server-side and returns the "best" (the one with '
        'the highest log probability per token). Results cannot be streamed.',
        default=None,
    )
    n: int = Field(description='The number of sequences to return.', default=None)
    num_return_sequences: int = Field(
        description='The number of sequences to return.', default=None
    )


class GenerateRequest(BaseRequest):
    prompt: str = Field(description='The prompt to generate from.')

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

        schema_extra = {
            'example': {
                'prompt': 'Hello, my name is',
                'id': '18d92585-7b66-4b7c-b818-71287c122c57',
                'num_beams': 5,
                'max_tokens': 50,
                'max_new_tokens': 50,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.95,
                'repetition_penalty': 1.0,
                'echo': False,
                'stop': ['\n', '.'],
                'stop_str': ['\n', '.'],
                'do_sample': True,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0,
                'best_of': 5,
                'logprobs': None,
                'n': 3,
                'num_return_sequences': 3,
            }
        }


class ChatRequest(BaseRequest):
    messages: List[dict] = Field(description='The prompt to generate from.')

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

        schema_extra = {
            'example': {
                'messages': [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                'id': '18d92585-7b66-4b7c-b818-71287c122c57',
                'num_beams': 5,
                'max_tokens': 50,
                'max_new_tokens': 50,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.95,
                'repetition_penalty': 1.0,
                'echo': False,
                'stop': ['\n', '.'],
                'stop_str': ['\n', '.'],
                'do_sample': True,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0,
                'best_of': 5,
                'logprobs': None,
                'n': 3,
                'num_return_sequences': 3,
            }
        }


class ResponseObjectEnum(str, Enum):
    GENERATION = 'text_completion'
    CHAT = 'chat.completion'


class BaseResponse(BaseModel):
    # session id
    id: str = Field(description='The session id of the generation.', default=None)

    object: ResponseObjectEnum = Field(
        description='The task type of the response.', default=None
    )
    created: int = Field(description='The timestamp of the response.', default=None)
    choices: List[dict] = Field(
        description='The generated text. It contains 5 keys: `index`, `text`, `message`, `logprobs`, '
        '`finish_reason`. For generation mode, `message` is None. For chat mode, '
        '`text` is None.'
    )
    prompt: str = Field(
        description='The prompt used to generate the response.', default=None
    )
    usage: dict = Field(
        description='The usage of the model. It contains 3 keys: `prompt_tokens`, '
        '`completion_tokens`, `total_tokens`. `prompt_tokens` is the length of input, '
        'in streaming mode this also includes the length of past_key_values. '
        '`completion_tokens` is the length of the generated text, in streaming mode this '
        'also includes the length of text generated in previous steps.'
        '`total_tokens` is the total length of the `prompt_tokens` and `completion_tokens`.'
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

        schema_extra = {
            'example': {
                'id': '18d92585-7b66-4b7c-b818-71287c122c57',
                'object': 'chat.completion',
                'create': 12345678,
                'choices': [
                    {
                        "index": 0,
                        "text": None,
                        "message": {
                            "role": "assistant",
                            "content": "\n\nHello there, how may I assist you today?",
                        },
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                'prompt': 'Hello there.',
                'usage': {
                    'prompt_tokens': 0,
                    'input_length': 10,
                    'completion_tokens': 10,
                    'total_tokens': 20,
                },
            }
        }
