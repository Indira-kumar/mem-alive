from mem_alive.embedding.local_embedding_provider import LocalEmbeddingProvider


async def test_local_embedding_provider_accepts_model_url_and_timeout():
    provider = LocalEmbeddingProvider(
        base_url="http://localhost:12345",
        model="embedding-model",
        timeout=321.0,
    )

    assert provider.url == "http://localhost:12345"
    assert provider.model == "embedding-model"
    assert provider.client.timeout.read == 321.0

    await provider.aclose()
