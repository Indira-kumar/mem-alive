from mem_alive.schema.memory_schema import Memory
from mem_alive.store.semantic_store import SemanticStore

VOCAB = ["the", "capital", "of", "france", "is", "paris", "lyon", "weather", "today"]


def make_store(backend, provider_factory, **overrides):
    provider = provider_factory(VOCAB)
    return SemanticStore(embedding_provider=provider, db=backend, **overrides)


async def test_remember_then_recall_roundtrip(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="the capital of france is paris", metadata={})

    results = await store.recall(namespace="ns", search_query="the capital of france is paris", metadata={})

    assert len(results) == 1
    assert results[0].content == "the capital of france is paris"


async def test_recall_filters_out_unrelated_memories(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="the capital of france is paris", metadata={})

    results = await store.recall(namespace="ns", search_query="weather today", metadata={})

    assert results == []


async def test_remember_supersedes_contradicting_fact(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider, contradiction_threshold=0.8)
    await store.remember(namespace="ns", fact="the capital of france is paris", metadata={})
    await store.remember(namespace="ns", fact="the capital of france is lyon", metadata={})

    results = await store.recall(namespace="ns", search_query="the capital of france is paris", metadata={})

    # the old (paris) fact should be superseded, only the new (lyon) one comes back
    assert len(results) == 1
    assert results[0].content == "the capital of france is lyon"


async def test_recall_excludes_other_memory_types(backend, make_embedding_provider):
    provider = make_embedding_provider(VOCAB)
    store = SemanticStore(embedding_provider=provider, db=backend)
    vector = (await provider.embed(["the capital of france is paris"]))[0]

    # inserted directly as an episodic memory, sharing namespace + vector with a semantic query
    episodic_memory = Memory(
        id="ep-1",
        content="the capital of france is paris",
        vector=vector,
        metadata={},
        memory_type="episodic",
        namespace="ns",
    )
    await backend.upsert(episodic_memory)

    results = await store.recall(namespace="ns", search_query="the capital of france is paris", metadata={})

    assert results == []


async def test_remember_does_not_supersede_other_memory_types(backend, make_embedding_provider):
    provider = make_embedding_provider(VOCAB)
    store = SemanticStore(embedding_provider=provider, db=backend, contradiction_threshold=0.5)
    vector = (await provider.embed(["the capital of france is paris"]))[0]

    episodic_memory = Memory(
        id="ep-1",
        content="the capital of france is paris",
        vector=vector,
        metadata={},
        memory_type="episodic",
        namespace="ns",
    )
    await backend.upsert(episodic_memory)

    await store.remember(namespace="ns", fact="the capital of france is paris", metadata={})

    untouched = await backend.get_memory_by_id(namespace="ns", id="ep-1")
    assert untouched.superseded is False


async def test_recall_respects_top_k(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider, contradiction_threshold=1.1)  # disable supersession
    await store.remember(namespace="ns", fact="the capital of france is paris", metadata={})
    await store.remember(namespace="ns", fact="the capital of france is lyon", metadata={})

    results = await store.recall(
        namespace="ns", search_query="the capital of france is paris", metadata={}, top_k=1
    )

    assert len(results) == 1
