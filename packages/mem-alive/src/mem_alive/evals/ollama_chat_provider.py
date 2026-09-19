import httpx

from .chat_provider import ChatProvider


class OllamaChatProvider(ChatProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
