"""Agent-memory lifecycle eval; scripted mode measures orchestration, not model quality.

Run ``python -m evals.agent_memory`` for deterministic correctness, or add
``--live --model MODEL --embedding-model MODEL`` for actual Ollama quality.
Live failures never fall back to scripted inference. Each run uses a fresh DB.
"""

import argparse
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mem_alive.backend.sqlite_backend import SQLiteBackend
from mem_alive.core.agent_memory import AgentMemory
from mem_alive.embedding.embedding_provider import EmbeddingProvider
from mem_alive.embedding.local_embedding_provider import LocalEmbeddingProvider
from mem_alive.inference import JSONInferenceProvider

INITIAL = "The billing service uses SQLite as its primary database."
COMPATIBLE = "The billing service uses Redis for its job queue."
CORRECTION = (
    "Correction: the billing service now uses PostgreSQL instead of SQLite as its primary database."
)
CURRENT = "The billing service uses PostgreSQL as its primary database."
UNRELATED = "What is the national flower of Japan?"


class FixedEmbedding(EmbeddingProvider):
    """Identical vectors ensure similarity alone cannot erase compatible facts."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if text == UNRELATED else [1.0, 0.0] for text in texts]


class ScriptedInference:
    """An explicit fixture policy: validates orchestration only, never model quality."""

    async def extract(self, messages, *, context=None):
        return [CURRENT if m["content"] == CORRECTION else m["content"] for m in messages]

    async def extract_entities(self, texts):
        return [
            [
                name
                for name in ("billing", "SQLite", "PostgreSQL", "Redis", "MySQL")
                if name.lower() in text.lower()
            ]
            for text in texts
        ]

    async def summarize(self, previous_summary, messages):
        return (previous_summary + " " + " ".join(m["content"] for m in messages)).strip()


async def evaluate(memory: AgentMemory, *, live: bool = False) -> dict[str, bool]:
    """Score independently observable state, event identity, and scoped retrieval."""

    async def add(text, namespace="billing"):
        return await memory.add([{"role": "user", "content": text}], namespace)

    results = {}
    await add(INITIAL)
    initial_records = await memory.list("billing")
    initial_id = next((r["id"] for r in initial_records if "sqlite" in r["content"].lower()), None)
    await add(COMPATIBLE)
    records = await memory.list("billing")
    results["compatible-facts"] = (
        len(records) == 2
        and any("sqlite" in r["content"].lower() for r in records)
        and any("redis" in r["content"].lower() for r in records)
    )
    events = await add(CORRECTION)
    records = await memory.list("billing")
    current = [r for r in records if "postgresql" in r["content"].lower()]
    results["correction-accumulates"] = (
        len(records) == 3
        and len(current) == 1
        and current[0]["id"] != initial_id
        and any(e["event"] == "ADD" for e in events)
        and any(r["id"] == initial_id and "sqlite" in r["content"].lower() for r in records)
    )
    events = await add(CURRENT)
    results["duplicate-noop"] = len(await memory.list("billing")) == 3 and all(
        e["event"] == "NONE" for e in events
    )
    await add("The payroll service uses MySQL as its database.", "payroll")
    hits = await memory.search("Which database does the billing service use?", "billing")
    results["namespace-isolation"] = bool(hits) and all(r["namespace"] == "billing" for r in hits)
    results["abstention"] = not await memory.search(UNRELATED, "billing", threshold=0.3)
    return results


async def run(
    *,
    live=False,
    model="qwen3:8b",
    embedding_model="embeddinggemma",
    base_url="http://localhost:11434",
) -> dict:
    """Return evidence with explicit mode and settings; provider failures propagate."""
    with tempfile.TemporaryDirectory(prefix="mem-alive-eval-") as directory:
        path = Path(directory) / "memory.sqlite3"
        embeddings = LocalEmbeddingProvider(base_url, embedding_model) if live else FixedEmbedding()
        inference = JSONInferenceProvider(model, base_url=base_url) if live else ScriptedInference()
        async with AgentMemory(SQLiteBackend(path), embeddings, inference) as memory:
            results = await evaluate(memory, live=live)
            before = await memory.list("billing")
            summary_before = await memory.db.get_summary("billing")
            messages_before = await memory.db.recent_messages("billing")
        async with SQLiteBackend(path) as reopened:
            after = await reopened.list_memories("billing")
            results["context-durability"] = (
                bool(summary_before)
                and bool(messages_before)
                and await reopened.get_summary("billing") == summary_before
                and await reopened.recent_messages("billing") == messages_before
            )
            results["restart-durability"] = {r["id"]: r["content"] for r in before} == {
                r.id: r.content for r in after
            } and len(after) == 3
    return {
        "mode": "live-model-quality" if live else "scripted-orchestration-only",
        "model": model if live else None,
        "embedding_model": embedding_model if live else None,
        "temperature": 0 if live else None,
        "abstention_threshold": 0.3,
        "date": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "passed": all(results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    settings = {
        "mode": "live-model-quality" if args.live else "scripted-orchestration-only",
        "model": args.model if args.live else None,
        "embedding_model": args.embedding_model if args.live else None,
        "date": datetime.now(timezone.utc).isoformat(),
    }
    try:
        report = asyncio.run(run(**vars(args)))
    except httpx.HTTPError as exc:
        print(
            json.dumps(
                {**settings, "passed": False, "failure": "environment", "error": type(exc).__name__}
            )
        )
        raise SystemExit(2) from None
    except ValueError as exc:
        print(
            json.dumps(
                {**settings, "passed": False, "failure": "invalid-model-output", "error": str(exc)}
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
