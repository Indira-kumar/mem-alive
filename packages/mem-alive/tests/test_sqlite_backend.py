import asyncio
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from mem_alive.backend.sqlite_backend import ConcurrentWriteError, SQLiteBackend
from mem_alive.schema.memory_schema import Memory


def memory(id="m", namespace="project", vector=None, **kwargs):
    return Memory(
        id,
        "Use PostgreSQL",
        vector if vector is not None else [1.0, 0.0],
        {"project": "billing"},
        "semantic",
        namespace,
        **kwargs,
    )


async def test_persistence_and_scoped_identity(tmp_path):
    path = tmp_path / "memory.db"
    first = memory()
    async with SQLiteBackend(path) as db:
        await db.upsert(first)
        await db.upsert(memory(namespace="other"))
    async with SQLiteBackend(path) as db:
        assert await db.get_memory_by_id("project", "m") == first
        await db.delete("other", "m")
        assert await db.get_memory_by_id("project", "m") == first
        assert await db.get_memory_by_id("other", "m") is None
        assert await db.history("missing", "m") == []


async def test_search_filters_ranking_and_pagination(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        await db.upsert(memory("a"))
        await db.upsert(memory("b", vector=[0, 1]))
        await db.upsert(replace(memory("c"), metadata={"project": "other"}))
        await db.upsert(memory("foreign", namespace="other"))
        results = await db.search("project", [1, 0], {"project": "billing"}, 10)
        assert [r.memory.id for r in results] == ["a", "b"]
        assert [r.score for r in results] == [1, 0]
        assert [m.id for m in await db.list_memories("project", limit=1, offset=1)] == ["b"]
        assert await db.search("project", [1, 0], {}, 0) == []
        assert await db.list_memories("project", {"missing": None}) == []
        assert await db.list_memories("project' OR 1=1 --") == []


async def test_atomic_dimension_failure_rolls_back_data_history_and_setting(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        with pytest.raises(ValueError, match="dimension"):
            await db.apply_changes([memory("a"), memory("b", vector=[1, 0, 0])], [])
        assert await db.list_memories("project") == []
        assert await db.history("project", "a") == []
        assert await db.revision("project") == 0
        await db.upsert(memory("c", vector=[1, 0, 0]))
        with pytest.raises(ValueError, match="dimension"):
            await db.search("project", [1, 0], {})
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        with pytest.raises(ValueError, match="dimension"):
            await db.upsert(memory())


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 0]])
async def test_invalid_embeddings_rejected(tmp_path, vector):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        with pytest.raises(ValueError):
            await db.upsert(memory(vector=vector))
        with pytest.raises(ValueError):
            await db.search("project", vector, {})
        assert await db.list_memories("project") == []


async def test_audit_and_optimistic_concurrency(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        original = memory()
        await db.upsert(original)
        updated = replace(original, content="Use SQLite", updated_at=datetime.now(timezone.utc))
        await db.apply_changes([updated], [], expected=[original])
        with pytest.raises(ConcurrentWriteError):
            await db.apply_changes([], [("project", "m")], expected=[original])
        assert await db.get_memory_by_id("project", "m") == updated
        await db.apply_changes([], [("project", "m")], expected=[updated])
        with pytest.raises(ConcurrentWriteError):
            await db.apply_changes([original], [], expected=[updated])
        events = await db.history("project", "m")
        assert [e["event"] for e in events] == ["ADD", "UPDATE", "DELETE"]
        assert events[0]["after"]["content"] == "Use PostgreSQL"
        assert events[-1]["before"]["content"] == "Use SQLite"
        assert "vector" not in events[0]["after"]
        await db.delete("project", "m")
        assert len(await db.history("project", "m")) == 3


async def test_parallel_connections_serialize_mutations(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as a, SQLiteBackend(path) as b:
        await asyncio.gather(*(db.upsert(memory(str(i))) for i, db in enumerate([a, b] * 5)))
        assert len(await a.list_memories("project")) == 10


async def test_closed_database_and_invalid_batch(tmp_path):
    db = SQLiteBackend(tmp_path / "memory.db")
    with pytest.raises(ValueError, match="once"):
        await db.apply_changes([memory()], [("project", "m")])
    with pytest.raises(ValueError):
        await db.list_memories("project", limit=-1)
    with pytest.raises(ValueError):
        await db.search("project", [1, 0], {}, -1)
    with pytest.raises(ValueError):
        await db.upsert(replace(memory(), metadata={"invalid": float("nan")}))
    with pytest.raises(ValueError):
        await db.upsert(replace(memory(), created_at=datetime(2020, 1, 1)))
    await db.aclose()
    await db.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await db.get_memory_by_id("project", "m")


async def test_namespace_revision_rejects_phantom_insert_between_connections(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as a, SQLiteBackend(path) as b:
        revision = await a.revision("project")
        assert revision == 0
        assert await a.list_memories("project") == []
        await b.upsert(memory("first"))
        with pytest.raises(ConcurrentWriteError, match="Namespace"):
            await a.apply_changes(
                [memory("duplicate")], [], namespace_revision=("project", revision)
            )
        assert [m.id for m in await a.list_memories("project")] == ["first"]
        assert await a.history("project", "duplicate") == []
        current = await a.revision("project")
        await b.upsert(memory("unrelated", namespace="other"))
        await a.apply_changes([memory("second")], [], namespace_revision=("project", current))
        assert await a.revision("project") > current
        deletion_revision = await a.revision("project")
        await b.delete("project", "first")
        with pytest.raises(ConcurrentWriteError):
            await a.apply_changes([], [], namespace_revision=("project", deletion_revision))


async def test_namespace_revision_survives_restart_and_failed_batch(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as db:
        await db.upsert(memory())
        revision = await db.revision("project")
        with pytest.raises(ValueError, match="dimension"):
            await db.apply_changes(
                [memory("a"), memory("b", vector=[1, 0, 0])],
                [],
                namespace_revision=("project", revision),
            )
        assert await db.revision("project") == revision
        assert [m.id for m in await db.list_memories("project")] == ["m"]
        assert await db.history("project", "a") == []
    async with SQLiteBackend(path) as db:
        assert await db.revision("project") == revision


def test_future_schema_version_is_rejected_without_changes(tmp_path):
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=3")
    with pytest.raises(ValueError, match="schema version"):
        SQLiteBackend(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


async def test_conversation_context_is_bounded_scoped_and_revisioned(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as db:
        assert await db.recent_messages("project") == []
        assert await db.get_summary("project") == ""
        messages = [{"role": "user", "content": f"message {i}"} for i in range(14)]
        await db.apply_changes(
            [], [], messages=messages, context_namespace="project", summary="User prefers Python."
        )
        assert await db.recent_messages("project") == messages[-10:]
        assert await db.get_summary("project") == "User prefers Python."
        revision = await db.revision("project")
        assert revision > 0
        await db.apply_changes([], [], context_namespace="project", summary="Updated summary.")
        assert await db.revision("project") > revision
        with pytest.raises(ConcurrentWriteError):
            await db.apply_changes(
                [],
                [],
                messages=[{"role": "user", "content": "stale"}],
                context_namespace="project",
                namespace_revision=("project", revision),
            )
        assert await db.recent_messages("other") == []
        assert await db.get_summary("other") == ""
        assert await db.revision("other") == 0
    async with SQLiteBackend(path) as db:
        assert await db.recent_messages("project") == messages[-10:]
        assert await db.get_summary("project") == "Updated summary."


async def test_entity_links_are_ranked_grouped_replaced_and_deleted(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as db:
        await db.apply_changes(
            [memory("a"), memory("b")],
            [],
            entities={
                "a": [("PostgreSQL", [1, 0]), ("Python", [0, 1])],
                "b": [("PostgreSQL", [1, 0])],
            },
        )
        await db.apply_changes(
            [memory("foreign", namespace="other")], [], entities={"foreign": [("private", [1, 0])]}
        )
        assert await db.entity_search("project", [1, 0]) == [
            ("PostgreSQL", 1.0, ["a", "b"]),
            ("Python", 0.0, ["a"]),
        ]
        assert await db.entity_search("project", [1, 0], top_k=1) == [
            ("PostgreSQL", 1.0, ["a", "b"]),
        ]
        await db.apply_changes(
            [replace(memory("a"), content="Use SQLite")], [], entities={"a": [("SQLite", [1, 0])]}
        )
        assert await db.entity_search("project", [1, 0]) == [
            ("PostgreSQL", 1.0, ["b"]),
            ("SQLite", 1.0, ["a"]),
        ]
        await db.delete("project", "b")
    async with SQLiteBackend(path) as db:
        assert await db.entity_search("project", [1, 0]) == [("SQLite", 1.0, ["a"])]
        assert await db.entity_search("other", [1, 0]) == [("private", 1.0, ["foreign"])]
        assert await db.entity_search("missing", [1, 0]) == []


async def test_auxiliary_writes_roll_back_with_failed_entity_dimension(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        with pytest.raises(ValueError, match="dimension"):
            await db.apply_changes(
                [memory()],
                [],
                messages=[{"role": "user", "content": "new preference"}],
                context_namespace="project",
                summary="new summary",
                entities={"m": [("PostgreSQL", [1, 0, 0])]},
            )
        assert await db.get_memory_by_id("project", "m") is None
        assert await db.history("project", "m") == []
        assert await db.recent_messages("project") == []
        assert await db.get_summary("project") == ""
        assert await db.revision("project") == 0
        assert await db.entity_search("project", [1, 0, 0]) == []
        await db.upsert(memory(vector=[1, 0, 0]))


async def test_v1_database_migrates_without_losing_history_or_dimension(tmp_path):
    path = tmp_path / "memory.db"
    original = memory()
    old_record = json.loads(SQLiteBackend._encode(original))
    old_record.pop("content_hash")
    old_record.pop("keywords")
    payload = json.dumps(old_record)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE memories(namespace TEXT,id TEXT,payload TEXT, PRIMARY KEY(namespace,id))"
        )
        connection.execute("CREATE TABLE settings(key TEXT PRIMARY KEY,value INTEGER)")
        connection.execute(
            "CREATE TABLE history(sequence INTEGER PRIMARY KEY, namespace TEXT, "
            "id TEXT,event TEXT,before_payload TEXT,after_payload TEXT,created_at TEXT)"
        )
        connection.execute("INSERT INTO memories VALUES ('project','m',?)", (payload,))
        connection.execute("INSERT INTO settings VALUES ('dimension',2)")
        connection.execute(
            "INSERT INTO history VALUES (7,'project','m','ADD',NULL,?,?)",
            (payload, original.created_at.isoformat()),
        )
        connection.execute("PRAGMA user_version=1")
    async with SQLiteBackend(path) as db:
        assert await db.get_memory_by_id("project", "m") == original
        assert await db.revision("project") == 7
        assert len(await db.history("project", "m")) == 1
        assert await db.get_summary("project") == ""
        assert await db.recent_messages("project") == []
        assert await db.entity_search("project", [1, 0]) == []
        await db.apply_changes([], [], context_namespace="project", summary="A migrated summary")
        assert await db.revision("project") == 8
        await db.apply_changes(
            [replace(original, content="corrected legacy fact")], [], expected=[original]
        )
        with pytest.raises(ValueError, match="dimension"):
            await db.upsert(memory(vector=[1, 0, 0]))
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


async def test_invalid_context_and_entities_do_not_write(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        with pytest.raises(ValueError, match="namespace"):
            await db.apply_changes([], [], messages=[])
        with pytest.raises(ValueError, match="role"):
            await db.apply_changes(
                [], [], messages=[{"role": "bad", "content": "x"}], context_namespace="project"
            )
        with pytest.raises(ValueError, match="content"):
            await db.apply_changes(
                [], [], messages=[{"role": "user", "content": ""}], context_namespace="project"
            )
        with pytest.raises(ValueError, match="upserted"):
            await db.apply_changes([], [], entities={"missing": [("Python", [1, 0])]})
        with pytest.raises(ValueError, match="unique"):
            await db.apply_changes(
                [memory()], [], entities={"m": [("Python", [1, 0]), ("Python", [1, 0])]}
            )
        with pytest.raises(ValueError, match="nonnegative"):
            await db.entity_search("project", [1, 0], -1)
        assert await db.revision("project") == 0


async def test_exact_hash_uniqueness_is_scoped_and_transactional(tmp_path):
    path = tmp_path / "memory.db"
    original = replace(memory(), content_hash="hash-one")
    async with SQLiteBackend(path) as db:
        await db.upsert(original)
        assert await db.find_by_hash("project", "hash-one") == original
        assert await db.find_by_hash("other", "hash-one") is None
        revision = await db.revision("project")
        with pytest.raises(sqlite3.IntegrityError):
            await db.apply_changes(
                [memory("would-rollback"), replace(original, id="duplicate")], []
            )
        assert await db.get_memory_by_id("project", "would-rollback") is None
        assert await db.history("project", "would-rollback") == []
        assert await db.revision("project") == revision
        await db.upsert(replace(original, namespace="other"))
        await db.upsert(memory("hashless-a"))
        await db.upsert(memory("hashless-b"))
        with pytest.raises(ValueError, match="nonempty"):
            await db.find_by_hash("project", "")
    async with SQLiteBackend(path) as db:
        assert await db.find_by_hash("project", "hash-one") == original


async def test_recent_memories_orders_updates_and_limits_namespace(tmp_path):
    async with SQLiteBackend(tmp_path / "memory.db") as db:
        old = replace(memory("a"), updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        recent = replace(memory("z"), updated_at=datetime(2021, 1, 1, tzinfo=timezone.utc))
        await db.apply_changes([old, recent, memory("foreign", namespace="other")], [])
        assert [m.id for m in await db.recent_memories("project", 1)] == ["z"]
        assert [m.id for m in await db.recent_memories("project")] == ["z", "a"]
        assert await db.recent_memories("project", 0) == []
        with pytest.raises(ValueError, match="nonnegative"):
            await db.recent_memories("project", -1)


async def test_embedding_model_identity_persists_and_rejects_same_dimension_model(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path, embedding_model="model-a") as db:
        await db.upsert(memory())
    with pytest.raises(ValueError, match="model differs"):
        SQLiteBackend(path, embedding_model="model-b")
    async with SQLiteBackend(path, embedding_model="model-a") as db:
        assert await db.get_memory_by_id("project", "m") is not None
    async with SQLiteBackend(path) as db:
        assert await db.get_memory_by_id("project", "m") is not None
    with pytest.raises(ValueError, match="nonempty"):
        SQLiteBackend(path, embedding_model="")


async def test_legacy_database_model_can_be_explicitly_bound_once(tmp_path):
    path = tmp_path / "memory.db"
    async with SQLiteBackend(path) as db:
        await db.upsert(memory())
    async with SQLiteBackend(path, embedding_model="legacy-model") as db:
        assert await db.get_memory_by_id("project", "m") is not None
    with pytest.raises(ValueError, match="model differs"):
        SQLiteBackend(path, embedding_model="another-model")
