from dataclasses import replace

import pytest

pytest.importorskip("lancedb")

from mem_alive.backend.lancedb_backend import LanceDBBackend
from mem_alive.schema.memory_schema import Memory


def make_memory(
    id: str,
    content: str = "production database postgres",
    vector: list[float] | None = None,
    metadata: dict | None = None,
    memory_type: str = "semantic",
    namespace: str = "ns",
) -> Memory:
    return Memory(
        id=id,
        content=content,
        vector=vector if vector is not None else [1.0, 0.0, 0.0],
        metadata=(
            metadata
            if metadata is not None
            else {"environment": "production", "owner": "platform"}
        ),
        memory_type=memory_type,
        namespace=namespace,
    )


async def test_lancedb_upsert_get_update_and_delete(tmp_path):
    async with LanceDBBackend(tmp_path / "lance") as backend:
        original = make_memory("memory-1")
        await backend.upsert(original)

        fetched = await backend.get_memory_by_id("ns", "memory-1")
        assert fetched == original

        updated = replace(original, content="production database postgresql")
        await backend.upsert(updated)
        assert await backend.get_memory_by_id("ns", "memory-1") == updated

        await backend.delete("ns", "memory-1")
        assert await backend.get_memory_by_id("ns", "memory-1") is None


async def test_lancedb_search_ranks_and_filters_without_crossing_namespace(tmp_path):
    async with LanceDBBackend(tmp_path / "lance") as backend:
        await backend.upsert(make_memory("exact"))
        await backend.upsert(
            make_memory("partial", vector=[0.8, 0.2, 0.0], metadata={"environment": "staging"})
        )
        await backend.upsert(make_memory("other-ns", namespace="other"))

        results = await backend.search(
            namespace="ns",
            vector=[1.0, 0.0, 0.0],
            metadata={"environment": "production"},
            top_k=5,
        )

        assert [result.memory.id for result in results] == ["exact"]
        assert results[0].score == pytest.approx(1.0)


async def test_lancedb_persists_across_backend_instances(tmp_path):
    uri = tmp_path / "lance"
    first = LanceDBBackend(uri)
    await first.upsert(make_memory("persistent"))
    await first.aclose()

    async with LanceDBBackend(uri) as reopened:
        fetched = await reopened.get_memory_by_id("ns", "persistent")

    assert fetched is not None
    assert fetched.content == "production database postgres"


async def test_lancedb_rejects_embedding_dimension_changes(tmp_path):
    async with LanceDBBackend(tmp_path / "lance") as backend:
        await backend.upsert(make_memory("first"))

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            await backend.upsert(make_memory("second", vector=[1.0, 0.0]))


async def test_lancedb_search_with_zero_top_k_returns_no_results(tmp_path):
    async with LanceDBBackend(tmp_path / "lance") as backend:
        await backend.upsert(make_memory("memory-1"))

        results = await backend.search("ns", [1.0, 0.0, 0.0], {}, top_k=0)

    assert results == []
