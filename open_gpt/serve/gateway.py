"""The serve module provides a simple way to serve a model using Jina."""
from typing import List

from jina import DocumentArray
from jina import Gateway as BaseGateway
from jina.serve.runtimes.servers.composite import CompositeServer
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description='The prompt to generate from.')

    # generation parameters
    num_beams: int = Field(description='The number of beams to use.', nullable=True)
    max_length: int = Field(
        description='The maximum length of the generated text.', nullable=True
    )
    temperature: float = Field(
        description='The temperature of the generation.', nullable=True
    )
    top_k: int = Field(description='The top k of the generation.', nullable=True)
    top_p: float = Field(description='The top p of the generation.', nullable=True)
    repetition_penalty: float = Field(
        description='The repetition penalty of the generation.', nullable=True
    )
    do_sample: bool = Field(
        description='Whether to sample from the generation.', nullable=True
    )
    num_return_sequences: int = Field(
        description='The number of sequences to return.', nullable=True
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

        schema_extra = {
            'example': {
                'prompt': 'Hello, my name is',
                'num_beams': 5,
                'max_length': 50,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.95,
                'repetition_penalty': 1.0,
                'do_sample': True,
                'num_return_sequences': 3,
            }
        }


class Gateway(BaseGateway, CompositeServer):
    """A simple Jina Gateway that can be used to serve a generation model."""

    def __init__(self, **kwargs):
        """Initialize a new Gateway."""

        super().__init__(**kwargs)

        self.grpc_gateway, self.http_gateway, *_ = self.servers

        from fastapi import Body, status
        from fastapi.responses import JSONResponse

        print(f'==> http: {self.http_gateway.app}')

        @self.http_gateway.app.post(path='/generate')
        async def generate(request: GenerateRequest = Body(...)):
            """Generate text from a prompt."""

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={'text': 'hello world'},
            )
