"""The serve module provides a simple way to serve a model using Jina."""
from typing import List

import jina
from jina import Document, DocumentArray
from jina import Gateway as BaseGateway
from jina.serve.runtimes.servers.composite import CompositeServer
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description='The prompt to generate from.')

    # generation parameters
    num_beams: int = Field(description='The number of beams to use.', default=None)
    max_length: int = Field(
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
    do_sample: bool = Field(
        description='Whether to sample from the generation.', default=None
    )
    num_return_sequences: int = Field(
        description='The number of sequences to return.', default=None
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

        from fastapi import Body, status
        from fastapi.responses import JSONResponse

        def _extend_rest_function(app):
            @app.api_route(path='/generate', methods=['POST'])
            async def generate(payload: GenerateRequest = Body(...)):
                """Generate text from a prompt."""

                parameters = payload.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                    exclude={'prompt'},
                )

                async for docs, error in self.streamer.stream(
                    docs=DocumentArray([Document(text=payload.prompt)]),
                    exec_endpoint='/generate',
                    parameters=parameters,
                ):
                    if error:
                        return JSONResponse(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={'message': error.name},
                        )
                    else:
                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content={
                                'generated_text': docs[0].tags.get('generated_text')
                            },
                        )

            return app

        jina.helper.extend_rest_interface = _extend_rest_function
