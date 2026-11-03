"""Regression for valid multi-turn input that exceeds a single query's size cap."""

import json

import httpx
from mem_alive.backend.sqlite_backend import SQLiteBackend
from mem_alive.core.agent_memory import AgentMemory
from mem_alive.embedding.embedding_provider import EmbeddingProvider
from mem_alive.inference import JSONInferenceProvider


async def test_long_turns_keep_full_extraction_input_and_bound_retrieval_window(tmp_path):
    class RecordingEmbedding(EmbeddingProvider):
        def __init__(self):
            self.inputs = []

        async def embed(self, texts):
            self.inputs.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    calls = []

    def respond(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][-1]["content"])
        calls.append(data)
        field = payload["format"]["required"][0]
        result = {
            "facts": {"facts": ["The project uses PostgreSQL."]},
            "summary": {"summary": "The project uses PostgreSQL."},
            "entities": {"entities": [["PostgreSQL"]]},
        }[field]
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(result)},
            },
        )

    embeddings = RecordingEmbedding()
    messages = [
        {"role": "user", "content": "a" * 9_000},
        {"role": "assistant", "content": "b" * 9_000},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = JSONInferenceProvider("test-model", client=client)
        async with AgentMemory(
            SQLiteBackend(tmp_path / "memory.db"), embeddings, provider
        ) as memory:
            # A populated namespace used to trigger query NER and reject valid input.
            await memory.add(
                [{"role": "user", "content": "Existing context"}], "project", infer=False
            )
            embeddings.inputs.clear()
            events = await memory.add(messages, "project")
            assert events[0]["memory"]["content"] == "The project uses PostgreSQL."
            assert embeddings.inputs[0] == [
                (messages[0]["content"] + "\n" + messages[1]["content"])[-16_000:]
            ]
            extraction = next(data for data in calls if "context" in data)
            assert extraction["messages"] == messages
            assert extraction["context"]["relevant_memories"] == ["Existing context"]
            assert [data["texts"] for data in calls if "texts" in data] == [
                ["The project uses PostgreSQL."]
            ]
            assert (await memory.db.recent_messages("project"))[-2:] == messages
