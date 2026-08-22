from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    id: str
    content: str
    vector: list[float]
    metadata: dict
    memory_type: str
    namespace: str
    superseded: bool
    created_at: datetime
    updated_at: datetime
