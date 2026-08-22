from mem_alive.store.procedural_store import ProceduralStore

VOCAB = ["restart", "the", "docker", "service", "kubernetes", "weather", "today"]


def make_store(backend, provider_factory, **overrides):
    provider = provider_factory(VOCAB)
    return ProceduralStore(embedding_provider=provider, db=backend, **overrides)


async def test_remember_then_recall_roundtrip(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="restart the docker service", metadata={})

    results = await store.recall(namespace="ns", search_query="restart the docker service", metadata={})

    assert len(results) == 1
    assert results[0].content == "restart the docker service"


async def test_recall_filters_out_unrelated_memories(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)
    await store.remember(namespace="ns", fact="restart the docker service", metadata={})

    results = await store.recall(namespace="ns", search_query="weather today", metadata={})

    assert results == []


def test_keyword_score_divides_by_query_tokens(backend, make_embedding_provider):
    store = make_store(backend, make_embedding_provider)

    # every query word appears in the content -> full score, regardless of content length
    full_match = store._keyword_score("restart docker", "restart the docker service")
    assert full_match == 1.0

    # only one of the two query words appears in the content
    partial_match = store._keyword_score("restart kubernetes", "restart the docker service")
    assert partial_match == 0.5

    # no query words appear in the content
    no_match = store._keyword_score("weather today", "restart the docker service")
    assert no_match == 0.0
