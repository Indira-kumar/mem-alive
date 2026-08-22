from mem_alive.schema.memory_schema import Memory


def make_memory(id, namespace="ns", vector=None, metadata=None, content="content"):
    return Memory(
        id=id,
        content=content,
        vector=vector or [1.0, 0.0, 0.0],
        metadata=metadata or {},
        memory_type="semantic",
        namespace=namespace,
    )


async def test_upsert_and_get_roundtrip(backend):
    memory = make_memory("m1")
    await backend.upsert(memory)
    fetched = await backend.get_memory_by_id(namespace="ns", id="m1")
    assert fetched == memory


async def test_get_missing_id_returns_none(backend):
    assert await backend.get_memory_by_id(namespace="ns", id="missing") is None


async def test_get_wrong_namespace_returns_none(backend):
    await backend.upsert(make_memory("m1", namespace="ns-a"))
    assert await backend.get_memory_by_id(namespace="ns-b", id="m1") is None


async def test_delete_removes_memory(backend):
    await backend.upsert(make_memory("m1"))
    await backend.delete(namespace="ns", id="m1")
    assert await backend.get_memory_by_id(namespace="ns", id="m1") is None


async def test_delete_missing_id_does_not_crash(backend):
    await backend.delete(namespace="ns", id="missing")


async def test_search_ranks_by_cosine_similarity(backend):
    await backend.upsert(make_memory("exact", vector=[1.0, 0.0, 0.0]))
    await backend.upsert(make_memory("orthogonal", vector=[0.0, 1.0, 0.0]))
    await backend.upsert(make_memory("opposite", vector=[-1.0, 0.0, 0.0]))

    results = await backend.search(namespace="ns", vector=[1.0, 0.0, 0.0], metadata={}, top_k=3)

    assert [r.memory.id for r in results] == ["exact", "orthogonal", "opposite"]
    assert results[0].score == 1.0


async def test_search_respects_top_k(backend):
    for i in range(5):
        await backend.upsert(make_memory(f"m{i}", vector=[1.0, 0.0, 0.0]))
    results = await backend.search(namespace="ns", vector=[1.0, 0.0, 0.0], metadata={}, top_k=2)
    assert len(results) == 2


async def test_search_respects_namespace_isolation(backend):
    await backend.upsert(make_memory("m1", namespace="ns-a"))
    results = await backend.search(namespace="ns-b", vector=[1.0, 0.0, 0.0], metadata={}, top_k=10)
    assert results == []


async def test_search_filters_by_metadata(backend):
    await backend.upsert(make_memory("match", metadata={"topic": "billing"}))
    await backend.upsert(make_memory("no-match", metadata={"topic": "auth"}))

    results = await backend.search(
        namespace="ns", vector=[1.0, 0.0, 0.0], metadata={"topic": "billing"}, top_k=10
    )

    assert [r.memory.id for r in results] == ["match"]


async def test_search_empty_namespace_returns_empty_list(backend):
    results = await backend.search(namespace="ns", vector=[1.0, 0.0, 0.0], metadata={}, top_k=10)
    assert results == []
