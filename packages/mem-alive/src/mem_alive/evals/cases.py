from .schema import EvalCase, MemorySeed


def default_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            name="semantic-production-database",
            memory_type="semantic",
            memories=(
                MemorySeed("semantic", "The production database is PostgreSQL."),
                MemorySeed("semantic", "The deployment region is ap-south-1."),
            ),
            query="What is the production database?",
            expected_context=frozenset({"The production database is PostgreSQL."}),
            expected_answer_terms=frozenset({"PostgreSQL"}),
            top_k=1,
        ),
        EvalCase(
            name="episodic-incident-recall",
            memory_type="episodic",
            memories=(
                MemorySeed(
                    "episodic",
                    "During incident INC-42, the cache was restarted at 14:30 UTC.",
                ),
                MemorySeed("episodic", "The release retrospective happened on Friday."),
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
            memories=(
                MemorySeed(
                    "procedural",
                    "To rotate the API key, update the secret store, restart the worker, "
                    "then verify health checks.",
                ),
                MemorySeed(
                    "procedural", "To clear the local cache, remove the cache directory."
                ),
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
            memories=(
                MemorySeed("semantic", "The production database is PostgreSQL."),
                MemorySeed("semantic", "The deployment region is ap-south-1."),
            ),
            query="What is the customer support phone number?",
            expected_context=frozenset(),
            expected_answer_terms=frozenset({"do not know"}),
            top_k=1,
        ),
    )
