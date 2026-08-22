from datetime import datetime, timedelta, timezone

from mem_alive.schema.memory_schema import Memory
from mem_alive.store.episodic_store import EpisodicStore

VOCAB = ["team", "meeting", "standup", "weather", "today"]


def make_store(backend, provider_factory, **overrides):
    provider = provider_factory(VOCAB)
    return EpisodicStore(embedding_provider=provider, db=backend, **overrides)


async def test_remember_then_recall_roundtrip(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="team meeting", metadata={})

    results = await store.recall(namespace="ns", search_query="team meeting", metadata={})

    assert len(results) == 1
    assert results[0].content == "team meeting"


async def test_recall_filters_out_unrelated_memories(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="team meeting", metadata={})

    results = await store.recall(namespace="ns", search_query="weather today", metadata={})

    assert results == []


async def test_recall_ranks_recent_memory_above_old_memory(backend, make_embedding_provider):
    provider = make_embedding_provider(VOCAB)
    store = EpisodicStore(embedding_provider=provider, db=backend, half_life_hours=36, recall_threshold=0.8)
    vector = (await provider.embed(["team meeting"]))[0]

    new_memory = Memory(
        id="new", content="team meeting", vector=vector, metadata={}, memory_type="episodic", namespace="ns"
    )
    old_memory = Memory(
        id="old",
        content="team meeting",
        vector=vector,
        metadata={},
        memory_type="episodic",
        namespace="ns",
        created_at=datetime.now(timezone.utc) - timedelta(hours=200),
    )
    await backend.upsert(new_memory)
    await backend.upsert(old_memory)

    results = await store.recall(namespace="ns", search_query="team meeting", metadata={})

    assert [m.id for m in results] == ["new", "old"]
