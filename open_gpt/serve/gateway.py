"""The serve module provides a simple way to serve a model using Jina."""

import logging

import jina
from jina import Document, DocumentArray
from jina import Gateway as BaseGateway
from jina.serve.runtimes.servers.composite import CompositeServer
from pydantic import BaseModel, Field

from open_gpt.models.session import SessionManager


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description='The prompt to generate from.')

    # session id
    session_id: str = Field(
        description='The session id of the generation.', default=None
    )

    # generation parameters
    num_beams: int = Field(description='The number of beams to use.', default=None)
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
                'session_id': '18d92585-7b66-4b7c-b818-71287c122c57',
                'num_beams': 5,
                'max_new_tokens': 50,
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
        session_manager = SessionManager()

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

                session_id = payload.session_id
                if session_id is None:
                    import uuid

                    session_id = str(uuid.uuid4())
                    logging.info(
                        f"===> session_id is None, creating new session: {session_id}"
                    )
                else:
                    logging.info(f"===> session_id: {session_id}")

                async for docs, error in self.streamer.stream(
                    docs=DocumentArray(
                        [
                            Document(
                                embedding=session_manager.get(session_id),
                                tags={'prompt': payload.prompt},
                            )
                        ]
                    ),
                    exec_endpoint='/generate',
                    parameters=parameters,
                ):
                    if error:
                        return JSONResponse(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={'message': error.name, 'session_id': session_id},
                        )
                    else:
                        session_manager.update(session_id, docs[0].embedding)
                        logging.info(
                            f"===> seq_length of {session_id}: {session_manager.get_context_length(session_id)}"
                        )

                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content={
                                'generated_text': docs[0].tags.get('generated_text'),
                                'session_id': session_id,
                            },
                        )

            return app

        jina.helper.extend_rest_interface = _extend_rest_function
