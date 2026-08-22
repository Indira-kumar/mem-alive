from mem_alive.core.memory import Memory

VOCAB = ["deploy", "runbook", "docker", "team", "standup", "capital", "france", "paris", "weather", "today"]


def make_client(backend, provider_factory):
    provider = provider_factory(VOCAB)
    return Memory(embedding_provider=provider, db=backend)


async def test_targeted_recall_only_hits_requested_store(backend, make_embedding_provider):
    client = make_client(backend, make_embedding_provider)
    await client.remember(memory_type="semantic", namespace="ns", fact="capital france paris", metadata={})
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
    await client.remember(memory_type="semantic", namespace="ns", fact="capital france paris", metadata={})

    results = await client.recall(namespace="ns", search_query="weather today", metadata={})

    assert results == []
