"""Restart eval: run with `uv run --package mem-alive python evals/durable_memory.py`."""

import asyncio
import tempfile
from pathlib import Path

from mem_alive import Memory, SQLiteBackend
from mem_alive.embedding.embedding_provider import EmbeddingProvider


class FixedEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def evaluate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.db"
        storage = SQLiteBackend(path)
        before = Memory(FixedEmbeddings(), storage)
        saved = await before.remember("semantic", "personal", "Prefers tea", {"source": "user"})
        await before.update("personal", saved.id, "Prefers coffee")
        backup_path = await storage.backup(Path(directory) / "backup.db")
        del before

        after = Memory(FixedEmbeddings(), SQLiteBackend(path))
        recovered = await after.get("personal", saved.id)
        backup_recovered = await Memory(FixedEmbeddings(), SQLiteBackend(backup_path)).get(
            "personal", saved.id
        )
        checks = {
            "record recovered": recovered is not None and recovered.content == "Prefers coffee",
            "metadata recovered": recovered is not None
            and recovered.metadata == {"source": "user"},
            "scope preserved": await after.get("other", saved.id) is None,
            "search after restart": [
                item.id
                for item in await after.recall("personal", "coffee", {}, memory_type="semantic")
            ]
            == [saved.id],
            "history after restart": [
                event.action for event in await after.history("personal", saved.id)
            ]
            == ["add", "update"],
            "backup recovered": backup_recovered is not None
            and backup_recovered.content == "Prefers coffee",
            "browse after restart": [record.id for record in await after.list("personal")]
            == [saved.id],
        }
        passed = sum(checks.values())
        print(f"durable_memory: {passed}/{len(checks)} checks passed")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'} {name}")
        if passed != len(checks):
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
