"""Local, schema-constrained conversation memory extraction with Ollama."""

import json
from collections.abc import Sequence

import httpx

from .extractor import ExtractedMemory, Message

_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic", "episodic", "procedural"],
                    },
                    "content": {"type": "string"},
                    "source_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["memory_type", "content", "source_indices"],
            },
        }
    },
    "required": ["memories"],
}

_INSTRUCTIONS = """You extract durable memories for the person speaking in USER messages.
SEMANTIC = facts, preferences, and decisions about that person or their work. "I prefer dark
mode" is semantic, never procedural. EPISODIC = a specific event that already happened.
PROCEDURAL = explicit repeatable steps for performing a task, not a preference.
The numbered transcript marks USER and ASSISTANT lines. Only USER lines are evidence.
Never extract a claim from an ASSISTANT line unless a later USER line explicitly confirms it.
If a USER correction follows an ASSISTANT guess, save only the USER correction.
source_indices must point only to USER lines that directly state the memory.
Return no memories for greetings, unanswered questions, speculation, secrets, or generic advice.
Make each memory standalone, concise, and faithful. Never invent missing details.
Examples:
USER[0]: "I prefer dark mode." -> semantic "The user prefers dark mode", source_indices [0].
USER[0]: "Yesterday I merged PR 42." -> episodic "The user merged PR 42", [0].
USER[0]: "When deploying, run tests then build."
-> procedural "Run tests then build when deploying", [0].
ASSISTANT[0]: "You prefer tea." USER[1]: "Actually, I prefer coffee."
-> semantic "The user prefers coffee", [1]; do not save tea.
Return only JSON matching the schema."""


class OllamaMemoryExtractor:
    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434",
        timeout: float = 90,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=url, timeout=timeout)

    async def extract(self, messages: Sequence[Message]) -> list[ExtractedMemory]:
        if not messages:
            return []
        for message in messages:
            if message["role"] not in ("user", "assistant") or not message["content"].strip():
                raise ValueError("messages must have a user or assistant role and nonempty content")
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": _SCHEMA,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": _INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": "\n".join(
                            f"{message['role'].upper()}[{index}]: "
                            f"{json.dumps(message['content'])}"
                            for index, message in enumerate(messages)
                        ),
                    },
                ],
            },
        )
        response.raise_for_status()
        payload = json.loads(response.json()["message"]["content"])
        raw = payload["memories"]
        if not isinstance(raw, list) or len(raw) > 8:
            raise ValueError("extractor returned an invalid memory list")
        extracted = []
        for item in raw:
            memory_type = item["memory_type"]
            content = item["content"].strip()
            indices = item["source_indices"]
            if (
                memory_type not in ("semantic", "episodic", "procedural")
                or not content
                or len(content) > 500
                or not isinstance(indices, list)
                or any(
                    not isinstance(index, int)
                    or index < 0
                    or index >= len(messages)
                    for index in indices
                )
            ):
                raise ValueError("extractor returned an unsupported memory or source")
            if not indices or any(messages[index]["role"] != "user" for index in indices):
                continue
            extracted.append(ExtractedMemory(memory_type, content, tuple(indices)))
        return extracted

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> "OllamaMemoryExtractor":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()
