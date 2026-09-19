from .schema import EvalCase, MemorySeed

_BACKGROUND_MEMORIES = (
    MemorySeed("semantic", "The analytics warehouse uses BigQuery."),
    MemorySeed("episodic", "The staging deployment completed at 18:10 UTC."),
    MemorySeed(
        "procedural",
        "To restore a backup, verify its checksum before importing the archive.",
    ),
    MemorySeed("semantic", "Customer invoices are retained for seven years."),
    MemorySeed("episodic", "The quarterly planning session ended at noon."),
    MemorySeed(
        "procedural",
        "To deploy the docs site, build the static bundle and invalidate the CDN.",
    ),
    MemorySeed("semantic", "The deployment region is ap-south-1."),
    MemorySeed(
        "episodic",
        "During incident INC-17, an expired certificate was renewed at 09:00 UTC.",
    ),
    MemorySeed(
        "procedural",
        "To rotate database credentials, update the vault and recycle connection pools.",
    ),
    MemorySeed("semantic", "The on-call rotation changes every Monday."),
    MemorySeed("episodic", "The release retrospective happened on Friday."),
    MemorySeed("procedural", "To clear the local cache, remove the cache directory."),
)


def _among_background(target: MemorySeed) -> tuple[MemorySeed, ...]:
    midpoint = len(_BACKGROUND_MEMORIES) // 2
    return _BACKGROUND_MEMORIES[:midpoint] + (target,) + _BACKGROUND_MEMORIES[midpoint:]


def default_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            name="semantic-production-database",
            memory_type="semantic",
            memories=_among_background(
                MemorySeed("semantic", "The production database is PostgreSQL.")
            ),
            query="What is the production database?",
            expected_context=frozenset({"The production database is PostgreSQL."}),
            expected_answer_terms=frozenset({"PostgreSQL"}),
            top_k=1,
        ),
        EvalCase(
            name="episodic-incident-recall",
            memory_type="episodic",
            memories=_among_background(
                MemorySeed(
                    "episodic",
                    "During incident INC-42, the cache was restarted at 14:30 UTC.",
                )
            ),
            query="What happened during incident INC-42, and when?",
            expected_context=frozenset(
                {"During incident INC-42, the cache was restarted at 14:30 UTC."}
            ),
            expected_answer_terms=frozenset({"cache", "14:30"}),
            top_k=1,
        ),
        EvalCase(
            name="procedural-key-rotation",
            memory_type="procedural",
            memories=_among_background(
                MemorySeed(
                    "procedural",
                    "To rotate the API key, update the secret store, restart the worker, "
                    "then verify health checks.",
                )
            ),
            query="How do I rotate the API key?",
            expected_context=frozenset(
                {
                    "To rotate the API key, update the secret store, restart the worker, "
                    "then verify health checks."
                }
            ),
            expected_answer_terms=frozenset({"secret store", "restart", "health checks"}),
            top_k=1,
        ),
        EvalCase(
            name="semantic-unanswerable-query",
            memory_type="semantic",
            memories=_BACKGROUND_MEMORIES,
            query="What is the customer support phone number?",
            expected_context=frozenset(),
            expected_answer_terms=frozenset({"do not know"}),
            top_k=1,
        ),
    )
