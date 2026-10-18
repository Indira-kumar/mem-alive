"""Deterministic retrieval eval: `uv run --package mem-alive python evals/retrieval.py`."""

import asyncio

from mem_alive import InMemoryBackend, Memory
from mem_alive.embedding.embedding_provider import EmbeddingProvider


class FixedEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def evaluate() -> None:
    memory = Memory(FixedEmbeddings(), InMemoryBackend())
    target = await memory.remember("semantic", "project", "Deploy with uv", {})
    for index in range(120):
        await memory.remember("episodic", "project", f"Event {index}", {})
    hits = await memory.recall_hits("project", "Deploy", {}, top_k=1, memory_type="semantic")
    all_hits = await memory.recall_hits("project", "Deploy", {}, top_k=3)
    checks = {
        "typed recall survives crowding": len(hits) == 1 and hits[0].memory.id == target.id,
        "global limit": len(all_hits) == 3,
        "ranked scores": all_hits == sorted(all_hits, key=lambda hit: (-hit.score, hit.memory.id)),
        "zero limit": await memory.recall("project", "Deploy", {}, top_k=0) == [],
        "limit above candidate budget": len(
            await memory.recall("project", "Deploy", {}, top_k=110, memory_type="episodic")
        )
        == 110,
    }
    passed = sum(checks.values())
    print(f"retrieval: {passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
