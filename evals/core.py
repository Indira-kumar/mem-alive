"""Fast, deterministic probes of the current public memory client.

The vectors isolate store policy from embedding-model quality. A failed probe is
reported rather than silently changing the expected behavior to match v0.1.
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field

from mem_alive.backend.in_memory_backend import InMemoryBackend
from mem_alive.core.memory import Memory
from mem_alive.embedding.embedding_provider import EmbeddingProvider


@dataclass(frozen=True)
class Seed:
    memory_type: str
    content: str
    vector: tuple[float, ...]
    namespace: str = "project-a"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Probe:
    name: str
    category: str
    seeds: tuple[Seed, ...]
    query: str
    query_vector: tuple[float, ...]
    required: frozenset[str]
    allowed: frozenset[str] = frozenset()
    min_results: int = 0
    max_results: int | None = None
    namespace: str = "project-a"
    metadata: dict[str, str] = field(default_factory=dict)
    memory_type: str | None = None
    top_k: int = 5


@dataclass(frozen=True)
class Result:
    name: str
    category: str
    passed: bool
    required_recall: float
    context_precision: float
    retrieved: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    limit_exceeded: bool


class FixedEmbedding(EmbeddingProvider):
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vectors[text]) for text in texts]


A = (1.0, 0.0, 0.0)
B = (0.0, 1.0, 0.0)


def probes() -> tuple[Probe, ...]:
    """Gold behavior, including known gaps in the current implementation."""
    return (
        Probe(
            "namespace-isolation",
            "scope",
            (
                Seed("semantic", "Project A uses PostgreSQL", A),
                Seed("semantic", "Project B uses SQLite", A, namespace="project-b"),
            ),
            "Which database does this project use?",
            A,
            frozenset({"Project A uses PostgreSQL"}),
            memory_type="semantic",
        ),
        Probe(
            "metadata-isolation",
            "scope",
            (
                Seed("semantic", "Team red deploys on Friday", A, metadata={"team": "red"}),
                Seed("semantic", "Team blue deploys on Tuesday", A, metadata={"team": "blue"}),
            ),
            "When does this team deploy?",
            A,
            frozenset({"Team red deploys on Friday"}),
            metadata={"team": "red"},
            memory_type="semantic",
        ),
        Probe(
            "semantic-version-visibility",
            "versioning",
            (
                Seed("semantic", "The service uses MySQL", A),
                Seed("semantic", "The service uses PostgreSQL", A),
            ),
            "Which database does the service use now?",
            A,
            frozenset({"The service uses PostgreSQL"}),
            memory_type="semantic",
        ),
        Probe(
            "similar-compatible-facts",
            "reconciliation",
            (
                Seed("semantic", "PostgreSQL is the primary database", A),
                Seed("semantic", "PostgreSQL is also used for analytics", A),
            ),
            "Where is PostgreSQL used?",
            A,
            frozenset(
                {"PostgreSQL is the primary database", "PostgreSQL is also used for analytics"}
            ),
            memory_type="semantic",
        ),
        Probe(
            "episodes-remain-distinct",
            "episode",
            (
                Seed("episodic", "Build 41 failed after a timeout", A),
                Seed("episodic", "Build 42 passed after retry", A),
            ),
            "What happened in the two builds?",
            A,
            frozenset({"Build 41 failed after a timeout", "Build 42 passed after retry"}),
            memory_type="episodic",
        ),
        Probe(
            "unrelated-query-abstains",
            "abstention",
            (Seed("semantic", "The database is PostgreSQL", A),),
            "What is the support phone number?",
            B,
            frozenset(),
            memory_type="semantic",
        ),
        Probe(
            "federated-top-k",
            "ranking",
            (
                Seed("semantic", "Use integration tests for this repository", A),
                Seed("episodic", "An integration test caught yesterday's bug", A),
                Seed("procedural", "Run integration tests before unit tests", A),
            ),
            "What testing context matters?",
            A,
            frozenset(),
            allowed=frozenset(
                {
                    "Use integration tests for this repository",
                    "An integration test caught yesterday's bug",
                    "Run integration tests before unit tests",
                }
            ),
            min_results=1,
            max_results=1,
            top_k=1,
        ),
    )


async def evaluate(probe: Probe) -> Result:
    vectors = {seed.content: seed.vector for seed in probe.seeds}
    vectors[probe.query] = probe.query_vector
    memory = Memory(FixedEmbedding(vectors), InMemoryBackend())
    for seed in probe.seeds:
        await memory.remember(seed.memory_type, seed.namespace, seed.content, seed.metadata)
    hits = await memory.recall(
        namespace=probe.namespace,
        search_query=probe.query,
        metadata=probe.metadata,
        memory_type=probe.memory_type,
        top_k=probe.top_k,
    )
    return score_retrieval(probe, tuple(hit.content for hit in hits))


def score_retrieval(probe: Probe, retrieved: tuple[str, ...]) -> Result:
    """Score exact context identity, including abstention and duplicate injection."""
    found = set(retrieved)
    missing = tuple(sorted(probe.required - found))
    relevant = probe.required | probe.allowed
    unexpected = tuple(sorted(found - relevant))
    if probe.required:
        recall = len(probe.required & found) / len(probe.required)
    elif probe.allowed:
        recall = float(len(retrieved) >= probe.min_results)
    else:
        recall = float(not found)
    precision = (
        len(relevant & found) / len(retrieved)
        if retrieved
        else float(not probe.required and probe.min_results == 0)
    )
    limit_exceeded = probe.max_results is not None and len(retrieved) > probe.max_results
    return Result(
        probe.name,
        probe.category,
        not missing
        and not unexpected
        and len(retrieved) == len(found)
        and len(retrieved) >= probe.min_results
        and not limit_exceeded,
        recall,
        precision,
        retrieved,
        missing,
        unexpected,
        limit_exceeded,
    )


async def run() -> tuple[Result, ...]:
    return tuple(await asyncio.gather(*(evaluate(probe) for probe in probes())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic memory behavior probes")
    parser.add_argument("--strict", action="store_true", help="fail if a product probe fails")
    args = parser.parse_args()
    results = asyncio.run(run())
    print(json.dumps({"results": [asdict(result) for result in results]}, indent=2))
    raise SystemExit(1 if args.strict and not all(result.passed for result in results) else 0)


if __name__ == "__main__":
    main()
