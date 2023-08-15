"""The serve module provides a simple way to serve a model using Jina."""

from datetime import datetime

import jina
from fastapi.encoders import jsonable_encoder
from jina import Document, DocumentArray
from jina import Gateway as BaseGateway
from jina.serve.runtimes.servers.composite import CompositeServer

from .pydantic_models import (
    BaseResponse,
    ChatRequest,
    GenerateRequest,
    ResponseObjectEnum,
)


class Gateway(BaseGateway, CompositeServer):
    """A simple Jina Gateway that can be used to serve a generation model."""

    def __init__(self, **kwargs):
        """Initialize a new Gateway."""

        super().__init__(**kwargs)

        from fastapi import Body, status
        from fastapi.responses import JSONResponse

        def _update_key(parameters):
            key_maps = {
                'max_tokens': 'max_new_tokens',
                'n': 'num_return_sequences',
                'stop': 'stop_str',
            }
            for openai_key, hf_key in key_maps.items():
                parameters[hf_key] = parameters.pop(openai_key) or parameters[hf_key]
            return parameters

        def _extend_rest_function(app):
            @app.api_route(path='/generate', methods=['POST'])
            @app.api_route(path='/codegen/completions', methods=['POST'])
            async def generate(payload: GenerateRequest = Body(...)):
                """Generate text from a prompt."""

                parameters = payload.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                    exclude={'prompt'},
                )

                parameters = _update_key(parameters)

                async for docs, error in self.streamer.stream(
                    docs=DocumentArray(
                        [
                            Document(
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
                            content={'message': error.name},
                        )
                    else:
                        _tags = docs[0].tags.copy()
                        _tags['usage'] = {k: int(v) for k, v in _tags['usage'].items()}

                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content=jsonable_encoder(
                                BaseResponse(
                                    **_tags,
                                    object=ResponseObjectEnum.GENERATION,
                                    created=int(datetime.now().timestamp())
                                )
                            ),
                        )

            @app.api_route(path='/generate_stream', methods=['POST'])
            async def generate_stream(payload: GenerateRequest = Body(...)):
                """Generate text from a prompt in streaming."""

                from fastapi import HTTPException
                from sse_starlette.sse import EventSourceResponse

                parameters = payload.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                    exclude={'prompt'},
                )

                parameters = _update_key(parameters)

                async def event_generator():
                    completion_tokens = 0

                    stop_flag = False
                    while not stop_flag:
                        parameters['completion_tokens'] = completion_tokens

                        async for docs, error in self.streamer.stream(
                            docs=input_docs,
                            exec_endpoint='/generate_stream',
                            parameters=parameters,
                        ):
                            if error:
                                # TODO: find best practice to handle errors in sse
                                raise HTTPException(status_code=500, detail=error)

                            input_docs[0].tags['input_ids'] = docs[0].tags.get(
                                'output_ids'
                            )
                            input_docs[0].blob = docs[0].blob

                            stop_flag = docs[0].tags.get('choices')[0].get(
                                'finish_reason'
                            ) in [
                                'stop',
                                'length',
                            ]
                            completion_tokens += 1

                            _tags = docs[0].tags.copy()
                            for k in ['input_ids', 'output_ids', 'past_key_values']:
                                _tags.pop(k) if k in _tags else None
                            _tags['usage'] = {
                                k: int(v) for k, v in _tags['usage'].items()
                            }
                            yield {
                                "data": jsonable_encoder(
                                    BaseResponse(
                                        **_tags,
                                        object=ResponseObjectEnum.GENERATION,
                                        created=int(datetime.now().timestamp())
                                    )
                                )
                            }

                input_docs = DocumentArray(
                    [
                        Document(
                            tags={'prompt': payload.prompt},
                        )
                    ]
                )

                return EventSourceResponse(event_generator())

            @app.api_route(path='/chat', methods=['POST'])
            async def chat(payload: ChatRequest = Body(...)):
                """Chat with a model."""
                parameters = payload.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                    exclude={'messages'},
                )

                parameters = _update_key(parameters)

                async for docs, error in self.streamer.stream(
                    docs=DocumentArray(
                        [
                            Document(
                                tags={'prompt': payload.messages},
                            )
                        ]
                    ),
                    exec_endpoint='/chat',
                    parameters=parameters,
                ):
                    if error:
                        return JSONResponse(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={'message': error.name},
                        )
                    else:
                        _tags = docs[0].tags.copy()
                        _tags['usage'] = {k: int(v) for k, v in _tags['usage'].items()}

                        return JSONResponse(
                            status_code=status.HTTP_200_OK,
                            content=jsonable_encoder(
                                BaseResponse(
                                    **_tags,
                                    object=ResponseObjectEnum.CHAT,
                                    created=int(datetime.now().timestamp())
                                )
                            ),
                        )

            @app.api_route(path='/chat_stream', methods=['POST'])
            async def chat_stream(payload: ChatRequest = Body(...)):
                """Generate chat response in streaming mode."""

                from fastapi import HTTPException
                from sse_starlette.sse import EventSourceResponse

                parameters = payload.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                    exclude={'messages'},
                )

                parameters = _update_key(parameters)

                async def event_generator():
                    completion_tokens = 0

                    stop_flag = False
                    while not stop_flag:
                        parameters['completion_tokens'] = completion_tokens

                        async for docs, error in self.streamer.stream(
                            docs=input_docs,
                            exec_endpoint='/chat_stream',
                            parameters=parameters,
                        ):
                            if error:
                                # TODO: find best practice to handle errors in sse
                                raise HTTPException(status_code=500, detail=error)

                            input_docs[0].tags['input_ids'] = docs[0].tags.get(
                                'output_ids'
                            )
                            input_docs[0].blob = docs[0].blob

                            stop_flag = docs[0].tags.get('choices')[0].get(
                                'finish_reason'
                            ) in [
                                'stop',
                                'length',
                            ]
                            completion_tokens += 1

                            _tags = docs[0].tags.copy()
                            for k in ['input_ids', 'output_ids', 'past_key_values']:
                                _tags.pop(k) if k in _tags else None
                            _tags['usage'] = {
                                k: int(v) for k, v in _tags['usage'].items()
                            }
                            yield {
                                "data": jsonable_encoder(
                                    BaseResponse(
                                        **_tags,
                                        object=ResponseObjectEnum.CHAT,
                                        created=int(datetime.now().timestamp())
                                    )
                                )
                            }

                input_docs = DocumentArray(
                    [
                        Document(
                            tags={'prompt': payload.messages},
                        )
                    ]
                )

                return EventSourceResponse(event_generator())

            return app

        jina.helper.extend_rest_interface = _extend_rest_function
