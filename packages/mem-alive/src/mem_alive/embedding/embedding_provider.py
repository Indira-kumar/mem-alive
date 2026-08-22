from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()

    async def __aenter__(self):
        return self

    async def aclose(self):
        pass  # no op, we will override only if the implementation holds resources
