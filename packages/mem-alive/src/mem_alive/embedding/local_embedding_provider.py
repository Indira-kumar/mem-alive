from .embedding_provider import EmbeddingProvider
import httpx

class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.url = 'http://localhost:11434'
        self.model = 'embeddinggemma'
        self.client = httpx.AsyncClient(
            base_url= self.url
        )
    
    async def embed(self, texts:list[str]) -> list[list[float]]:
        response = await self.client.post(
            "/api/embed",
            json={
                'model':self.model,
                'input':texts
            }
        )
        response = response.json()['embeddings']
        return response

    async def aclose(self, exc_type, exc_value, traceback):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self):
        await self.aclose()