"""Deterministic local-database durability eval: ``python -m evals.sqlite``.

Fixed embeddings isolate the persistence contract from model quality. The task
saves a coding preference, restarts the process-facing client, corrects it, and
checks that another project cannot read or alter that memory.
"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from mem_alive.backend.sqlite_backend import ConcurrentWriteError, SQLiteBackend
from mem_alive.schema.memory_schema import Memory


async def evaluate(path: Path) -> dict[str, bool]:
    original = Memory(
        id="test-command",
        content="Run pytest before committing.",
        vector=[1.0, 0.0],
        metadata={"repository": "billing"},
        memory_type="semantic",
        namespace="billing",
    )
    async with SQLiteBackend(path) as db:
        await db.upsert(original)
    async with SQLiteBackend(path) as db:
        restored = await db.get_memory_by_id("billing", original.id)
        durability = restored == original
        isolation = await db.search("unrelated", [1, 0], {}, 10) == []
        await db.delete("unrelated", original.id)
        isolation &= await db.get_memory_by_id("billing", original.id) == original
        corrected = replace(original, content="Run uv run pytest before committing.")
        await db.apply_changes([corrected], [], expected=[original])
        results = await db.search("billing", [1, 0], {}, 10)
        correction = [r.memory.content for r in results] == [corrected.content]
        try:
            await db.apply_changes([], [("billing", original.id)], expected=[original])
        except ConcurrentWriteError:
            stale_write = await db.get_memory_by_id("billing", original.id) == corrected
        else:
            stale_write = False
        history = await db.history("billing", original.id)
        audit = [entry["event"] for entry in history] == ["ADD", "UPDATE"]
        revision = await db.revision("billing")
        async with SQLiteBackend(path) as other_client:
            captured = replace(original, id="editor", content="Use Vim.")
            await other_client.upsert(captured)
        try:
            await db.apply_changes(
                [replace(captured, id="duplicate-editor")],
                [],
                namespace_revision=("billing", revision),
            )
        except ConcurrentWriteError:
            concurrent_capture = await db.get_memory_by_id("billing", "duplicate-editor") is None
        else:
            concurrent_capture = False
        messages = [{"role": "user", "content": f"decision {i}"} for i in range(12)]
        await db.apply_changes(
            [corrected],
            [],
            messages=messages,
            context_namespace="billing",
            summary="The user runs tests before committing.",
            entities={corrected.id: [("pytest", [1, 0])]},
        )
    async with SQLiteBackend(path) as db:
        context = (
            await db.recent_messages("billing") == messages[-10:]
            and await db.get_summary("billing") == "The user runs tests before committing."
            and await db.recent_messages("unrelated") == []
            and await db.get_summary("unrelated") == ""
        )
        entity_links = (
            await db.entity_search("billing", [1, 0]) == [("pytest", 1.0, [corrected.id])]
            and await db.entity_search("unrelated", [1, 0]) == []
        )
        await db.delete("billing", corrected.id)
        entity_cleanup = await db.entity_search("billing", [1, 0]) == []
    return {
        "restart_durability": durability,
        "project_isolation": isolation,
        "corrected_preference_only": correction,
        "stale_write_rejected": stale_write,
        "correction_audit": audit,
        "concurrent_capture_rejected": concurrent_capture,
        "bounded_context_persistence_and_isolation": context,
        "entity_link_persistence_and_isolation": entity_links,
        "deleted_memory_entity_cleanup": entity_cleanup,
    }


def main() -> None:
    with TemporaryDirectory(prefix="mem-alive-sqlite-eval-") as directory:
        cases = asyncio.run(evaluate(Path(directory) / "memories.db"))
    print(json.dumps({"cases": cases, "passed": sum(cases.values()), "total": len(cases)}))
    raise SystemExit(0 if all(cases.values()) else 1)


if __name__ == "__main__":
    main()
