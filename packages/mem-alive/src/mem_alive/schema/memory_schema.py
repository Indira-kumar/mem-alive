from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    id: str
    content: str
    vector: list[float]
    metadata: dict
    memory_type: str
    namespace: str
    superseded: bool = False
    content_hash: str = ""
    keywords: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchResult:
    memory: Memory
    score: float
