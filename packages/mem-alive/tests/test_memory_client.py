import pytest
from mem_alive.core.memory import Memory

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


async def test_explicit_update_refreshes_content_and_vector(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    saved = await client.remember("semantic", "ns", "capital france paris", {"source": "user"})

    updated = await client.update("ns", saved.id, "capital france lyon")

    assert updated.id == saved.id
    assert updated.created_at == saved.created_at
    assert updated.updated_at >= saved.updated_at
    assert updated.vector != saved.vector
    assert (await client.get("ns", saved.id)).content == "capital france lyon"
    recalled = await client.recall("ns", "capital france lyon", {}, memory_type="semantic")
    assert recalled[0].id == saved.id


async def test_update_and_forget_are_namespace_scoped(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    saved = await client.remember("semantic", "private", "capital france paris", {})

    assert await client.get("other", saved.id) is None
    with pytest.raises(KeyError):
        await client.update("other", saved.id, "capital france lyon")
    await client.forget("other", saved.id)
    assert await client.get("private", saved.id) == saved
    await client.forget("private", saved.id)
    assert await client.get("private", saved.id) is None
    assert await client.history("private", saved.id) == []


async def test_delete_retains_audit_history(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    saved = await client.remember("semantic", "private", "capital france paris", {})
    await client.delete("private", saved.id)
    assert [event.action for event in await client.history("private", saved.id)] == [
        "add",
        "delete",
    ]


async def test_federated_recall_applies_global_limit_and_exposes_scores(
    backend, make_embedding_provider
):
    client = make_client(backend, make_embedding_provider)
    fact = "capital france paris"
    for memory_type in ("semantic", "episodic", "procedural"):
        await client.remember(memory_type, "ns", fact, {})

    hits = await client.recall_hits("ns", fact, {}, top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert [hit.memory for hit in hits] == await client.recall("ns", fact, {}, top_k=2)
    assert await client.recall("ns", fact, {}, top_k=0) == []


async def test_list_browses_only_requested_namespace(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    await client.remember("semantic", "personal", "capital france paris", {"topic": "travel"})
    await client.remember("procedural", "other", "deploy runbook docker", {})

    listed = await client.list("personal", metadata={"topic": "travel"})
    assert len(listed) == 1
    assert listed[0].namespace == "personal"


async def test_rejects_blank_memory_and_query(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    with pytest.raises(ValueError, match="nonempty"):
        await client.remember("semantic", "ns", " ")
    with pytest.raises(ValueError, match="nonempty"):
        await client.recall("ns", " ")
    with pytest.raises(ValueError, match="unsupported"):
        await client.remember("unknown", "ns", "fact")


async def test_recall_limit_can_exceed_default_candidate_budget(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    for _ in range(110):
        await client.remember("episodic", "ns", "team standup", {})

    assert len(await client.recall("ns", "team standup", {}, top_k=105)) == 105
