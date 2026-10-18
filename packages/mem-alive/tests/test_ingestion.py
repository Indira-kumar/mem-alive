import json
from hashlib import sha256

import httpx
import pytest
from mem_alive import Memory
from mem_alive.ingestion import ExtractedMemory, OllamaMemoryExtractor


class FakeExtractor:
    def __init__(self, memories):
        self.memories = memories

    async def extract(self, messages):
        return self.memories


async def test_ingest_retains_evidence_and_skips_exact_duplicates(backend, make_embedding_provider):
    memory = Memory(make_embedding_provider(["python", "prefer", "run", "tests", "uv"]), backend)
    messages = [
        {"role": "user", "content": "I prefer Python."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Run tests with uv."},
    ]
    extractor = FakeExtractor(
        [
            ExtractedMemory("semantic", "prefer python", (0,)),
            ExtractedMemory("semantic", "prefer python", (0,)),
            ExtractedMemory("procedural", "run tests uv", (2,)),
        ]
    )

    first = await memory.ingest("ns", messages, extractor, {"conversation_id": "c1"})
    second = await memory.ingest("ns", messages, extractor, {"conversation_id": "c1"})

    assert len(first) == 2
    assert second == []
    assert first[0].metadata["_source"] == [
        {"index": 0, "sha256": sha256(b"I prefer Python.").hexdigest()}
    ]
    assert first[1].metadata["conversation_id"] == "c1"


async def test_ingest_can_opt_in_to_source_text(backend, make_embedding_provider):
    memory = Memory(make_embedding_provider(["prefer", "python"]), backend)
    messages = [{"role": "user", "content": "I prefer Python."}]
    extractor = FakeExtractor([ExtractedMemory("semantic", "prefer python", (0,))])

    saved = await memory.ingest("ns", messages, extractor, store_source_text=True)

    assert saved[0].metadata["_source"][0]["text"] == "I prefer Python."


async def test_ingest_rejects_assistant_only_claim_before_writing(backend, make_embedding_provider):
    memory = Memory(make_embedding_provider(["python"]), backend)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "You prefer Python"},
    ]
    extractor = FakeExtractor([ExtractedMemory("semantic", "prefer python", (1,))])

    with pytest.raises(ValueError, match="source"):
        await memory.ingest("ns", messages, extractor)
    assert await memory.list("ns") == []


async def test_ollama_extractor_parses_schema_constrained_response():
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["format"]["properties"]["memories"]["type"] == "array"
        assert payload["options"]["temperature"] == 0
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "memories": [
                                {
                                    "memory_type": "semantic",
                                    "content": "Indira prefers Python",
                                    "source_indices": [0],
                                }
                            ]
                        }
                    )
                }
            },
        )

    async with httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(respond)
    ) as client:
        async with OllamaMemoryExtractor(client=client) as extractor:
            extracted = await extractor.extract([{"role": "user", "content": "I prefer Python"}])
    assert extracted == [ExtractedMemory("semantic", "Indira prefers Python", (0,))]


async def test_ollama_extractor_discards_assistant_only_claim():
    async with httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "memories": [
                                    {
                                        "memory_type": "semantic",
                                        "content": "Unsupported",
                                        "source_indices": [1],
                                    }
                                ]
                            }
                        )
                    }
                },
            )
        ),
    ) as client:
        async with OllamaMemoryExtractor(client=client) as extractor:
            extracted = await extractor.extract(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Unsupported"},
                ]
            )
            assert extracted == []
