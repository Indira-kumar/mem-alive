import hashlib

import pytest
from mem_alive.backend.sqlite_backend import SQLiteBackend
from mem_alive.core.agent_memory import AgentMemory
from mem_alive.embedding.embedding_provider import EmbeddingProvider

CURRENT = "The billing service uses PostgreSQL as its primary database."


class FixedEmbedding(EmbeddingProvider):
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ScriptedInference:
    def __init__(self):
        self.contexts = []

    async def extract(self, messages, *, context=None):
        self.contexts.append(context)
        return [m["content"] for m in messages]

    async def extract_entities(self, texts):
        return [[] for text in texts]

    async def summarize(self, previous_summary, messages):
        return (previous_summary + " " + " ".join(m["content"] for m in messages)).strip()


@pytest.fixture
async def memory(tmp_path):
    async with AgentMemory(
        SQLiteBackend(tmp_path / "memory.sqlite3"), FixedEmbedding(), ScriptedInference()
    ) as instance:
        yield instance


async def test_exact_duplicate_noop_and_corrections_accumulate(memory):
    first = await memory.add([{"role": "user", "content": "Use SQLite"}], "project")
    repeated = await memory.add([{"role": "user", "content": "Use SQLite"}], "project")
    correction = await memory.add([{"role": "user", "content": "Now use PostgreSQL"}], "project")
    assert repeated[0]["event"] == "NONE"
    assert repeated[0]["memory"]["id"] == first[0]["memory"]["id"]
    assert correction[0]["event"] == "ADD"
    assert len(await memory.list("project")) == 2
    assert len(await memory.history(first[0]["memory"]["id"], "project")) == 1


async def test_context_is_persisted_and_scoped(memory):
    await memory.add([{"role": "user", "content": "Private secret"}], "private")
    await memory.add([{"role": "user", "content": CURRENT}], "project")
    await memory.add([{"role": "user", "content": "It handles billing"}], "project")
    context = memory.inference.contexts[-1]
    assert "Private secret" not in str(context)
    assert CURRENT in context["summary"]
    assert CURRENT in context["relevant_memories"]
    assert CURRENT in context["recent_memories"]
    assert context["recent_messages"] == [{"role": "user", "content": CURRENT}]
    assert "current_date" in context


@pytest.mark.parametrize("failure", ["exception", "wrong_count"])
async def test_failed_entity_extraction_does_not_write_memory_or_context(memory, failure):
    class FailingEntities(ScriptedInference):
        async def extract_entities(self, texts):
            if failure == "exception":
                raise RuntimeError("model unavailable")
            return []

    memory.inference = FailingEntities()
    with pytest.raises((RuntimeError, ValueError)):
        await memory.add([{"role": "user", "content": CURRENT}], "project")
    assert await memory.list("project") == []
    assert await memory.db.get_summary("project") == ""
    assert await memory.db.recent_messages("project") == []
    assert await memory.db.revision("project") == 0


async def test_embedding_failure_during_commit_has_no_partial_writes(memory):
    class FailingEmbedding(FixedEmbedding):
        calls = 0

        async def embed(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("model unavailable")
            return await super().embed(texts)

    memory.embeddings = FailingEmbedding()
    with pytest.raises(RuntimeError, match="unavailable"):
        await memory.add([{"role": "user", "content": "Keep this fact"}], "project")
    assert await memory.list("project") == []
    assert await memory.db.get_summary("project") == ""
    assert await memory.db.recent_messages("project") == []


async def test_explicit_update_delete_and_history_are_scoped(memory):
    added = await memory.add([{"role": "user", "content": CURRENT}], "project", infer=False)
    id = added[0]["memory"]["id"]
    assert await memory.get(id, "other") is None
    with pytest.raises(KeyError):
        await memory.update(id, "stolen", "other")
    await memory.delete(id, "other")
    assert await memory.history(id, "other") == []
    updated = await memory.update(id, "Use MySQL", "project")
    assert updated["id"] == id
    assert updated["content"] == "Use MySQL"
    assert updated["content_hash"] == hashlib.sha256(b"Use MySQL").hexdigest()
    assert "mysql" in updated["keywords"]
    await memory.delete(id, "project")
    assert await memory.get(id, "project") is None
    history = await memory.history(id, "project")
    assert [entry["event"] for entry in history] == ["ADD", "UPDATE", "DELETE"]
    assert history[0]["after"]["content"] == CURRENT
    assert history[-1]["before"]["content"] == "Use MySQL"
    assert all("vector" not in entry["before"] for entry in history if entry["before"])


async def test_raw_mode_skips_all_language_model_calls(memory):
    class UnavailableInference:
        def __getattr__(self, name):
            raise AssertionError(f"Raw add must not call {name}")

    memory.inference = UnavailableInference()
    try:
        events = await memory.add([{"role": "user", "content": "Raw text"}], "project", infer=False)
        assert events[0]["event"] == "ADD"
    finally:
        memory.inference = ScriptedInference()


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[float("nan"), 1.0]]])
async def test_invalid_embeddings_fail_without_writes(memory, vectors):
    class InvalidEmbedding(FixedEmbedding):
        async def embed(self, texts):
            return vectors

    memory.embeddings = InvalidEmbedding()
    with pytest.raises(ValueError, match="embedding"):
        await memory.add([{"role": "user", "content": "fact"}], "project")
    assert await memory.list("project") == []


async def test_updated_content_deduplicates_to_existing_id(memory):
    first = await memory.add([{"role": "user", "content": "Use SQLite"}], "project", infer=False)
    id = first[0]["memory"]["id"]
    await memory.update(id, "Use PostgreSQL", "project")
    duplicate = await memory.add(
        [{"role": "user", "content": "Use PostgreSQL"}], "project", infer=False
    )
    assert duplicate[0]["event"] == "NONE"
    assert duplicate[0]["memory"]["id"] == id
    assert len(await memory.list("project")) == 1
    assert [item["event"] for item in await memory.history(id, "project")] == ["ADD", "UPDATE"]
