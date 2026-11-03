"""Deterministic hybrid-ranking quality eval; no live embedding/model claims."""

import json

from mem_alive.retrieval import rerank
from mem_alive.schema.memory_schema import Memory, SearchResult


def candidate(memory_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(Memory(memory_id, content, [1.0], {}, "semantic", "project"), score)


def evaluate() -> dict:
    corpus = [
        candidate("test-command", "Run tests with uv run pytest", 0.8),
        candidate("deploy-command", "Deploy application using the release workflow", 0.8),
        candidate("billing", "BillingService owns recurring customer invoices", 0.8),
    ]
    lexical = rerank("running tests", corpus, [])
    entities = rerank(
        "",
        corpus,
        [
            ("BillingService", 0.95, ["billing"]),
            ("application", 0.95, ["deploy-command", "test-command", "outside-candidates"]),
            ("phantom", 1, ["outside-candidates"]),
        ],
    )
    entity_parts = {hit.memory.id: parts for hit, parts in entities}
    checks = {
        "lemmatized_task_recall": lexical[0][0].memory.id == "test-command",
        "specific_entity_outranks_generic": entities[0][0].memory.id == "billing",
        "verified_entity_fanout_formula": (
            abs(entity_parts["billing"]["entity_boost"] - 0.475) < 1e-12
            and abs(entity_parts["deploy-command"]["entity_boost"] - 0.475 / 1.004) < 1e-12
        ),
        "candidate_boundary": {hit.memory.id for hit, _ in entities}
        == {hit.memory.id for hit in corpus},
        "bounded_explainable_scores": all(
            0 <= parts["score"] <= 1 and 0 <= parts["entity_boost"] <= 0.5
            for _, parts in lexical + entities
        ),
    }
    return {"kind": "deterministic-hybrid-eval", "checks": checks, "passed": all(checks.values())}


def main() -> None:
    result = evaluate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
