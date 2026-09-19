from mem_alive.schema.memory_schema import Memory, SearchResult


def make_memory(
    id: str = "id-1",
    content: str = "hello world",
    vector: list[float] | None = None,
    metadata: dict | None = None,
    memory_type: str = "semantic",
    namespace: str = "ns",
) -> Memory:
    return Memory(
        id=id,
        content=content,
        vector=vector or [0.1, 0.2],
        metadata=metadata or {},
        memory_type=memory_type,
        namespace=namespace,
    )


def test_memory_defaults():
    memory = make_memory()
    assert memory.superseded is False
    assert memory.created_at.tzinfo is not None
    assert memory.updated_at.tzinfo is not None


def test_memory_created_at_not_shared_across_instances():
    # regression test: created_at must use default_factory, not a shared eager default
    first = make_memory()
    second = make_memory()
    assert first.created_at is not second.created_at


def test_search_result_holds_memory_and_score():
    memory = make_memory()
    result = SearchResult(memory=memory, score=0.42)
    assert result.memory is memory
    assert result.score == 0.42
