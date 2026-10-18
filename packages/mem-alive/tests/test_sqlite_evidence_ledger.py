from dataclasses import replace
from datetime import datetime, timezone

import pytest
from mem_alive import Evidence, SourceEvent, SQLiteEvidenceLedger


def source() -> SourceEvent:
    return SourceEvent(
        id="turn-1",
        namespace="project-a",
        source_type="user_message",
        content="Run integration tests first.",
        occurred_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
        session_id="session-1",
    )


def evidence() -> Evidence:
    return Evidence(
        id="evidence-1",
        namespace="project-a",
        source_event_id="turn-1",
        kind="user_assertion",
        content="Integration tests should run first",
        span_start=0,
        span_end=28,
    )


def test_sources_and_evidence_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    ledger = SQLiteEvidenceLedger(path)
    event = source()
    item = evidence()

    assert ledger.append_source(event)
    assert ledger.add_evidence(item)

    restarted = SQLiteEvidenceLedger(path)
    assert restarted.get_source(event.namespace, event.id) == event
    assert restarted.get_evidence(item.namespace, item.id) == item
    assert restarted.evidence_for_source(event.namespace, event.id) == [item]


def test_identical_replay_is_no_op_and_conflicting_id_is_rejected(tmp_path):
    ledger = SQLiteEvidenceLedger(tmp_path / "memory.db")
    assert ledger.append_source(source())
    assert not ledger.append_source(source())
    with pytest.raises(ValueError, match="different content"):
        ledger.append_source(replace(source(), content="Use unit tests instead."))

    assert ledger.add_evidence(evidence())
    assert not ledger.add_evidence(evidence())
    with pytest.raises(ValueError, match="different content"):
        ledger.add_evidence(replace(evidence(), content="A different claim"))
    assert ledger.evidence_for_source("project-a", "turn-1") == [evidence()]


def test_evidence_requires_an_existing_source_and_valid_span(tmp_path):
    ledger = SQLiteEvidenceLedger(tmp_path / "memory.db")
    with pytest.raises(ValueError, match="unknown source"):
        ledger.add_evidence(evidence())

    ledger.append_source(source())
    for invalid in (
        replace(evidence(), span_start=0, span_end=None),
        replace(evidence(), span_start=-1),
        replace(evidence(), span_end=100),
        replace(evidence(), span_start=5, span_end=5),
    ):
        with pytest.raises(ValueError, match="span"):
            ledger.add_evidence(invalid)
    assert ledger.evidence_for_source("project-a", "turn-1") == []


def test_same_ids_in_different_namespaces_stay_isolated(tmp_path):
    ledger = SQLiteEvidenceLedger(tmp_path / "memory.db")
    other = replace(source(), namespace="project-b", content="Use unit tests first.")
    assert ledger.append_source(source())
    assert ledger.append_source(other)
    assert ledger.add_evidence(evidence())
    assert ledger.get_source("project-b", "turn-1") == other
    assert ledger.get_evidence("project-b", "evidence-1") is None
    with pytest.raises(ValueError, match="unknown source"):
        ledger.add_evidence(replace(evidence(), namespace="project-c"))


def test_source_requires_timezone(tmp_path):
    ledger = SQLiteEvidenceLedger(tmp_path / "memory.db")
    with pytest.raises(ValueError, match="timezone"):
        ledger.append_source(replace(source(), occurred_at=datetime(2026, 10, 18)))
