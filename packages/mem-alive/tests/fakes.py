from mem_alive.embedding.embedding_provider import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words embedding, no network calls.

    Cosine similarity between two texts reflects their shared-word overlap,
    which keeps recall/threshold tests meaningful without needing a real model.
    """

    def __init__(self, vocabulary: list[str]):
        self._vocabulary = vocabulary

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            tokens = set(text.lower().split())
            vectors.append([1.0 if word in tokens else 0.0 for word in self._vocabulary])
        return vectors
