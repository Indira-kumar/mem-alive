"""The embedding boundary rejects upstream failures before memory can be written."""

import httpx
import pytest
from mem_alive import LocalEmbeddingProvider


async def test_configured_model_endpoint_and_external_client_lifetime():
    def respond(request):
        assert request.url == "http://model/api/embed"
        assert b'"model":"local-embed"' in request.content
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = LocalEmbeddingProvider("http://model", "local-embed", client=client)
        assert await provider.embed(["a fact"]) == [[1.0, 0.0]]
        await provider.aclose()
        assert not client.is_closed


async def test_http_failure_is_not_treated_as_an_embedding_response():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "model unavailable"})
        )
    ) as client:
        provider = LocalEmbeddingProvider(client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed(["fact"])


@pytest.mark.parametrize(
    "vectors", [[], [[]], [[0.0, 0.0]], [[True, 1.0]], [["1", 2]], [[1, 2], [1]], [[1, 2], [2, 3]]]
)
async def test_malformed_vectors_cannot_reach_storage(vectors):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": vectors})
        )
    ) as client:
        provider = LocalEmbeddingProvider(client=client)
        with pytest.raises(ValueError):
            await provider.embed(["one fact"])
