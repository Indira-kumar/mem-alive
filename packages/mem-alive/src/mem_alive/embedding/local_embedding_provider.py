import httpx

from .embedding_provider import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "embeddinggemma",
    ):
        self.url = base_url
        self.model = model
        self.client = httpx.AsyncClient(base_url=self.url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.post("/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    async def aclose(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()
