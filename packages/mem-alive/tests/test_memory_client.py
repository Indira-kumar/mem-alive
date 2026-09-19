import pytest
from mem_alive.core.memory import Memory
from mem_alive.embedding.embedding_provider import EmbeddingProvider

VOCAB = [
    "deploy",
    "runbook",
    "docker",
    "team",
    "standup",
    "capital",
    "france",
    "paris",
    "weather",
    "today",
]


class CalibratedEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "stored memory": [1.0, 0.0],
            "related query": [0.65, 0.759934],
        }
        return [vectors[text] for text in texts]


def make_client(backend, provider_factory):
    provider = provider_factory(VOCAB)
    return Memory(embedding_provider=provider, db=backend)


async def test_targeted_recall_only_hits_requested_store(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    await client.remember(
        memory_type="semantic", namespace="ns", fact="capital france paris", metadata={}
    )
    await client.remember(memory_type="episodic", namespace="ns", fact="team standup", metadata={})

    results = await client.recall(
        namespace="ns", search_query="capital france paris", memory_type="semantic", metadata={}
    )

    assert len(results) == 1
    assert results[0].content == "capital france paris"


async def test_federated_recall_merges_all_store_types(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    # same content stored under all three types: isolates the gather/merge logic
    # from each store's own similarity math (already covered in their own test files)
    fact = "capital france paris"
    await client.remember(memory_type="semantic", namespace="ns", fact=fact, metadata={})
    await client.remember(memory_type="episodic", namespace="ns", fact=fact, metadata={})
    await client.remember(memory_type="procedural", namespace="ns", fact=fact, metadata={})

    results = await client.recall(namespace="ns", search_query=fact, metadata={})

    assert len(results) == 3
    assert {m.memory_type for m in results} == {"semantic", "episodic", "procedural"}


async def test_federated_recall_excludes_unrelated_memories(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    await client.remember(
        memory_type="semantic", namespace="ns", fact="capital france paris", metadata={}
    )

    results = await client.recall(namespace="ns", search_query="weather today", metadata={})

    assert results == []


async def test_default_threshold_recalls_moderately_similar_memory(backend):
    client = Memory(embedding_provider=CalibratedEmbeddingProvider(), db=backend)

    for memory_type in ("semantic", "episodic", "procedural"):
        namespace = f"ns-{memory_type}"
        await client.remember(memory_type, namespace, "stored memory", {})
        results = await client.recall(
            namespace=namespace,
            search_query="related query",
            memory_type=memory_type,
            metadata={},
        )
        assert [memory.content for memory in results] == ["stored memory"]


async def test_memory_client_accepts_stricter_recall_threshold(backend):
    client = Memory(
        embedding_provider=CalibratedEmbeddingProvider(),
        db=backend,
        recall_threshold=0.8,
    )
    await client.remember("semantic", "ns", "stored memory", {})

    results = await client.recall(
        namespace="ns",
        search_query="related query",
        memory_type="semantic",
        metadata={},
    )

    assert results == []


def test_memory_client_rejects_invalid_recall_threshold(backend):
    with pytest.raises(ValueError, match="recall_threshold must be between -1 and 1"):
        Memory(
            embedding_provider=CalibratedEmbeddingProvider(),
            db=backend,
            recall_threshold=1.1,
        )
