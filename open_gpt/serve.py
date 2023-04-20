"""The serve module provides a simple way to serve a model using Jina."""

from jina import DocumentArray, Gateway


class SSEGateway(Gateway):
    """A simple SSE gateway that can be used to stream events to the client."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def setup_server(self):
        """Setup the server."""
        self.app = self._get_app()

        @self.app.get("/generate")
        async def generate(request):
            """Generate text."""

            async def generate_text():
                """Generate text."""
                async for message in self._generate(request):
                    yield message

            return await self._stream(generate_text())

    async def _generate(self, request):
        """Generate text."""
        docs = DocumentArray([{"text": request.args["text"][0]}])
        await self._post("/generate", docs)
        return docs
