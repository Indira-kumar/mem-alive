import sqlite3
import stat

import pytest
from mem_alive import Memory as MemoryClient
from mem_alive import SQLiteBackend
from mem_alive.schema.memory_schema import Memory

from .fakes import FakeEmbeddingProvider


async def test_restart_recovers_memory_and_explicit_correction(tmp_path):
    path = tmp_path / "memory.db"
    first = MemoryClient(FakeEmbeddingProvider(["prefers", "python", "rust"]), SQLiteBackend(path))
    saved = await first.remember("semantic", "person", "prefers python", {"source": "user"})
    await first.update("person", saved.id, "prefers rust")

    reopened = MemoryClient(
        FakeEmbeddingProvider(["prefers", "python", "rust"]), SQLiteBackend(path)
    )
    restored = await reopened.get("person", saved.id)
    assert restored is not None
    assert restored.content == "prefers rust"
    assert restored.metadata == {"source": "user"}
    assert restored.created_at == saved.created_at
    recalled = await reopened.recall("person", "prefers rust", {}, memory_type="semantic")
    assert [m.id for m in recalled] == [saved.id]


async def test_namespace_and_metadata_filters_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    backend = SQLiteBackend(path)
    cases = [("a", "same", "billing"), ("b", "same", "billing"), ("a", "other", "auth")]
    for namespace, id, topic in cases:
        await backend.upsert(
            Memory(
                id=id,
                namespace=namespace,
                content=topic,
                vector=[1.0, 0.0],
                metadata={"topic": topic},
                memory_type="semantic",
            )
        )

    reopened = SQLiteBackend(path)
    matches = await reopened.search("a", [1.0, 0.0], {"topic": "billing"})
    assert [result.memory.id for result in matches] == ["same"]
    await reopened.delete("a", "same")
    assert await reopened.get_memory_by_id("a", "same") is None
    assert await reopened.get_memory_by_id("b", "same") is not None


async def test_history_survives_update_delete_and_restart(tmp_path):
    path = tmp_path / "memory.db"
    client = MemoryClient(FakeEmbeddingProvider(["tea", "coffee"]), SQLiteBackend(path))
    saved = await client.remember("semantic", "person", "tea", {})
    await client.update("person", saved.id, "coffee")
    await client.delete("person", saved.id)

    reopened = MemoryClient(FakeEmbeddingProvider(["tea", "coffee"]), SQLiteBackend(path))
    events = await reopened.history("person", saved.id)
    assert [event.action for event in events] == ["add", "update", "delete"]
    assert [event.memory.content for event in events] == ["tea", "coffee", "coffee"]
    assert await reopened.history("other", saved.id) == []
    await reopened.forget("person", saved.id)
    assert await SQLiteBackend(path).history("person", saved.id) == []


async def test_backup_is_readable_and_does_not_overwrite(tmp_path):
    backend = SQLiteBackend(tmp_path / "memory.db")
    saved = Memory("id", "content", [1.0, 0.0], {}, "semantic", "ns")
    await backend.upsert(saved)
    backup_path = await backend.backup(tmp_path / "backup.db")

    assert await SQLiteBackend(backup_path).get_memory_by_id("ns", "id") == saved
    assert stat.S_IMODE(backend.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        await backend.backup(backup_path)


async def test_list_filters_and_paginates(tmp_path):
    backend = SQLiteBackend(tmp_path / "memory.db")
    for id, topic, kind in [
        ("first", "billing", "semantic"),
        ("second", "billing", "semantic"),
        ("third", "auth", "procedural"),
    ]:
        await backend.upsert(Memory(id, topic, [1.0, 0.0], {"topic": topic}, kind, "ns"))
    assert len(await backend.list_memories("ns", {"topic": "billing"})) == 2
    assert len(await backend.list_memories("ns", {}, "semantic", limit=1, offset=1)) == 1
    assert await backend.list_memories("other", {}) == []


async def test_zero_vector_and_dimension_validation(tmp_path):
    backend = SQLiteBackend(tmp_path / "memory.db")
    await backend.upsert(Memory("zero", "empty", [0.0, 0.0], {}, "semantic", "ns"))
    assert (await backend.search("ns", [0.0, 0.0], {}))[0].score == 0.0
    with pytest.raises(ValueError, match="dimension"):
        await backend.search("ns", [1.0], {})


def test_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="version"):
        SQLiteBackend(path)
