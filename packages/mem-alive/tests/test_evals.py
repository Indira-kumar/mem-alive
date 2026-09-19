from mem_alive.backend.in_memory_backend import InMemoryBackend
from mem_alive.core.memory import Memory
from mem_alive.embedding.embedding_provider import EmbeddingProvider
from mem_alive.evals import ChatProvider, EvalRunner, RagAgent, default_cases


class ScenarioEmbeddingProvider(EmbeddingProvider):
    """Maps each built-in scenario to a deterministic orthogonal vector."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        signals = (
            "production database" in lowered or "postgresql" in lowered,
            "deployment region" in lowered or "ap-south-1" in lowered,
            "inc-42" in lowered,
            "retrospective" in lowered,
            "rotate the api key" in lowered or "secret store" in lowered,
            "local cache" in lowered or "cache directory" in lowered,
        )
        return [float(signal) for signal in signals]


class ContextChatProvider(ChatProvider):
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt


async def test_default_evals_cover_and_pass_all_memory_types():
    backend = InMemoryBackend()
    memory = Memory(embedding_provider=ScenarioEmbeddingProvider(), db=backend)
    agent = RagAgent(memory=memory, chat_provider=ContextChatProvider())

    report = await EvalRunner(memory=memory, agent=agent).run(default_cases())

    assert report.passed is True
    assert report.pass_rate == 1.0
    assert report.mean_context_recall == 1.0
    assert report.mean_context_precision == 1.0
    assert {result.memory_type for result in report.results} == {
        "semantic",
        "episodic",
        "procedural",
    }


async def test_rag_agent_marks_retrieved_memories_as_evidence():
    backend = InMemoryBackend()
    memory = Memory(embedding_provider=ScenarioEmbeddingProvider(), db=backend)
    chat = ContextChatProvider()
    agent = RagAgent(memory=memory, chat_provider=chat)
    await memory.remember(
        memory_type="semantic",
        namespace="ns",
        fact="The production database is PostgreSQL.",
        metadata={},
    )

    response = await agent.answer(
        namespace="ns",
        query="What is the production database?",
        memory_type="semantic",
    )

    assert response.context[0].content == "The production database is PostgreSQL."
    assert "Memories:\n- The production database is PostgreSQL." in response.answer
