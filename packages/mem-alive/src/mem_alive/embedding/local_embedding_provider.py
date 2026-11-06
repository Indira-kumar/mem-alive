import math

import httpx

from .embedding_provider import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Ollama embeddings with explicit timeouts and validated model responses."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "embeddinggemma",
        timeout: float = 120,
        client: httpx.AsyncClient | None = None,
    ):
        self.url = base_url
        self.model = model
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.post(
            f"{self.url.rstrip('/')}/api/embed", json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("embedding response does not match input count")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding vectors must be nonempty lists")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in vector
            ) or not any(vector):
                raise ValueError("embedding vectors must be finite and nonzero")
            if dimension is not None and dimension != len(vector):
                raise ValueError("embedding dimensions differ")
            dimension = len(vector)
        return vectors

    async def aclose(self):
        if self._owns_client:
            await self.client.aclose()
