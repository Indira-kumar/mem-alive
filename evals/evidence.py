"""Feature eval: source/evidence provenance survives replay and restart."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mem_alive import Evidence, SourceEvent, SQLiteEvidenceLedger


@dataclass(frozen=True)
class Result:
    passed: bool
    resolved_links: int
    expected_links: int
    replay_duplicates: int
    conflicting_replay_rejected: bool


def run() -> Result:
    """Exercise two kinds of evidence through a fresh database connection."""
    now = datetime(2026, 10, 18, tzinfo=timezone.utc)
    events = (
        SourceEvent("user-1", "project-a", "user_message", "Use integration tests.", now),
        SourceEvent("tool-1", "project-a", "tool_result", "Integration tests passed.", now),
    )
    items = (
        Evidence(
            "evidence-user", "project-a", "user-1", "user_assertion", "Use integration tests", 0, 21
        ),
        Evidence(
            "evidence-tool", "project-a", "tool-1", "tool_observation", "Tests passed", 0, 25
        ),
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.db"
        ledger = SQLiteEvidenceLedger(path)
        for event in events:
            ledger.append_source(event)
        for item in items:
            ledger.add_evidence(item)

        restarted = SQLiteEvidenceLedger(path)
        resolved = sum(
            restarted.get_evidence(item.namespace, item.id) == item
            and restarted.get_source(item.namespace, item.source_event_id)
            == next(event for event in events if event.id == item.source_event_id)
            for item in items
        )
        duplicates = sum(not restarted.append_source(event) for event in events) + sum(
            not restarted.add_evidence(item) for item in items
        )
        try:
            restarted.append_source(
                SourceEvent("user-1", "project-a", "user_message", "Changed", now)
            )
        except ValueError:
            conflict_rejected = True
        else:
            conflict_rejected = False
        return Result(
            passed=resolved == len(items)
            and duplicates == len(events) + len(items)
            and conflict_rejected,
            resolved_links=resolved,
            expected_links=len(items),
            replay_duplicates=duplicates,
            conflicting_replay_rejected=conflict_rejected,
        )


def main() -> None:
    result = run()
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
