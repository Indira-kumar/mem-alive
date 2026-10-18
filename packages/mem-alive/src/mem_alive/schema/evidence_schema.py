"""Source material and evidence selected from it."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SourceEvent:
    id: str
    namespace: str
    source_type: str
    content: str
    occurred_at: datetime
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    namespace: str
    source_event_id: str
    kind: str
    content: str
    span_start: int | None = None
    span_end: int | None = None
