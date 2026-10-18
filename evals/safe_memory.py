"""Deterministic safety eval: run with `uv run --package mem-alive python evals/safe_memory.py`."""

import asyncio

from mem_alive import InMemoryBackend, Memory
from mem_alive.embedding.embedding_provider import EmbeddingProvider


class CloseEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Deliberately identical: similarity cannot decide whether facts conflict.
        return [[1.0, 0.0] for _ in texts]


async def evaluate() -> None:
    memory = Memory(embedding_provider=CloseEmbeddings(), db=InMemoryBackend())
    original = await memory.remember("semantic", "person", "Indira prefers Python", {})
    compatible = await memory.remember("semantic", "person", "Indira also uses Rust", {})
    another_scope = await memory.remember("semantic", "project", "Project uses Python", {})

    checks = {
        "similar facts survive": {
            item.id for item in await memory.recall("person", "Python", {}, memory_type="semantic")
        }
        == {original.id, compatible.id},
        "explicit correction": (
            await memory.update("person", original.id, "Indira prefers Rust")
        ).id
        == original.id,
        "namespace isolation": await memory.get("person", another_scope.id) is None,
    }
    passed = sum(checks.values())
    print(f"safe_memory: {passed}/{len(checks)} checks passed")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
