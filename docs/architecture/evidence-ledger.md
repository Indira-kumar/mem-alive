# Source and evidence ledger

The first durable core slice records source material and evidence selected from it. It does not extract claims, create memories, or alter the existing `Memory` client. Later ingestion features can use these records without trusting a memory as its own evidence.

`SourceEvent` holds the original text, source type, session ID, namespace, and timezone-aware event time. `Evidence` holds a normalized claim or observation, its kind, and the ID of its source. Optional `span_start` and `span_end` locate supporting text using Python string offsets; omitting both means the whole source. Evidence content may paraphrase the source, so it is not required to equal the selected substring.

```python
from datetime import datetime, timezone
from mem_alive import Evidence, SourceEvent, SQLiteEvidenceLedger

ledger = SQLiteEvidenceLedger("memory.db")
ledger.append_source(
    SourceEvent(
        id="session-1:turn-4",
        namespace="project-a",
        source_type="user_message",
        content="Run integration tests first.",
        occurred_at=datetime.now(timezone.utc),
        session_id="session-1",
    )
)
ledger.add_evidence(
    Evidence(
        id="session-1:turn-4:claim-1",
        namespace="project-a",
        source_event_id="session-1:turn-4",
        kind="user_assertion",
        content="Integration tests should run first",
    )
)
```

The caller supplies stable IDs. Replaying an identical record returns `False`; reusing its ID with changed content raises `ValueError`. A new record returns `True`. A source must exist before evidence can refer to it. All reads require a namespace, and identical IDs in different namespaces remain separate. Each operation uses a SQLite transaction and its own connection so records remain available after process restart.

The public methods are `append_source`, `add_evidence`, `get_source`, `get_evidence`, and `evidence_for_source`. They are synchronous so a future host adapter can make a short durable write before scheduling extraction. This feature does not yet provide a background worker, write deadline, redaction, deletion, memory versioning, or automatic context delivery. Host adapters must treat source text as potentially sensitive and untrusted.

The feature eval is `python -m evals.evidence`. It checks provenance resolution after restart, replay without duplicate records, and rejection of a conflicting replay. Unit tests cover missing sources, invalid spans, timezone requirements, and namespace isolation. The broader core contracts in `evals/core_contracts.json` remain future targets because derived memory states are not part of this slice.
