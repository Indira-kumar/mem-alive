"""Live Ollama retrieval eval: `uv run --package mem-alive python evals/live_embeddings.py`."""

import asyncio
from datetime import datetime, timezone

from mem_alive import InMemoryBackend, LocalEmbeddingProvider, Memory


async def evaluate() -> None:
    facts = [
        ("semantic", "Indira prefers Python for automation."),
        ("semantic", "The mem-alive project uses uv for dependency management."),
        ("semantic", "The mem-alive repository is MIT licensed."),
        ("semantic", "The backup cadence is daily."),
        ("semantic", "Project Northstar escalation color is amber."),
        ("semantic", "The staging environment runs in Frankfurt."),
        ("procedural", "To run tests, execute uv run pytest."),
    ]
    questions = [
        ("Which tool manages dependencies for mem-alive?", 1),
        ("What license is mem-alive?", 2),
        ("Where does staging run?", 5),
        ("How do I run tests?", 6),
        ("What color is Project Northstar escalation?", 4),
    ]
    async with LocalEmbeddingProvider(timeout=90) as embedder:
        backend = InMemoryBackend()
        memory = Memory(embedder, backend)
        ids = [
            (await memory.remember(memory_type, "eval", fact, {})).id
            for memory_type, fact in facts
        ]
        top_3_hits = 0
        top_1_hits = 0
        for question, expected_index in questions:
            recalled = await memory.recall("eval", question, {}, top_k=3)
            recalled_ids = [item.id for item in recalled]
            top_3_hits += ids[expected_index] in recalled_ids
            top_1_hits += bool(recalled_ids) and recalled_ids[0] == ids[expected_index]
            vector = (await embedder.embed([question]))[0]
            raw = await backend.search("eval", vector, {}, top_k=3)
            print(
                f"query={question!r} matches={[item.content for item in recalled]} "
                f"raw={[(round(hit.score, 3), hit.memory.content) for hit in raw]}"
            )
        negative_questions = [
            "What is the monthly revenue?",
            "Which cloud provider hosts staging?",
            "What time of day should I run tests?",
        ]
        negative_results = []
        for question in negative_questions:
            recalled = await memory.recall("eval", question, {}, top_k=3)
            vector = (await embedder.embed([question]))[0]
            raw = await backend.search("eval", vector, {}, top_k=3)
            negative_results.append((question, recalled, raw))
    print(
        f"embeddinggemma date={datetime.now(timezone.utc).date()}: "
        f"top1={top_1_hits}/{len(questions)} "
        f"top3={top_3_hits}/{len(questions)} "
        f"negative_empty={all(not recalled for _, recalled, _ in negative_results)}"
    )
    for question, recalled, raw in negative_results:
        print(
            f"negative={question!r} matches={[item.content for item in recalled]} "
            f"raw={[(round(hit.score, 3), hit.memory.content) for hit in raw]}"
        )
    if top_3_hits < 4 or any(recalled for _, recalled, _ in negative_results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
