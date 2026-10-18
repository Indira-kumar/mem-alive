import httpx
import pytest
from mem_alive.embedding.local_embedding_provider import LocalEmbeddingProvider


async def test_embed_checks_status_and_batch_shape():
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    async with httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(respond)
    ) as client:
        async with LocalEmbeddingProvider(client=client) as provider:
            assert await provider.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
            assert await provider.embed([]) == []


async def test_embed_raises_for_bad_response():
    async with httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": []})
        )
    ) as client:
        async with LocalEmbeddingProvider(client=client) as provider:
            with pytest.raises(ValueError, match="invalid batch"):
                await provider.embed(["one"])
