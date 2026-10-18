import math

import httpx

from .embedding_provider import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "embeddinggemma",
        timeout: float = 60,
        client: httpx.AsyncClient | None = None,
    ):
        self.url = url
        self.model = model
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=url, timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.post("/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("embedding provider returned an invalid batch")
        dimension = None
        for vector in vectors:
            if (
                not isinstance(vector, list)
                or not vector
                or (dimension is not None and len(vector) != dimension)
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError("embedding provider returned an invalid batch")
            dimension = len(vector)
        return vectors

    async def aclose(self):
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()
