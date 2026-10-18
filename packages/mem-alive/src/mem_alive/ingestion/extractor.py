"""Boundary between conversation analysis and memory storage."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Message = Mapping[str, str]


@dataclass(frozen=True)
class ExtractedMemory:
    memory_type: Literal["semantic", "episodic", "procedural"]
    content: str
    source_indices: tuple[int, ...]


class MemoryExtractor(Protocol):
    async def extract(self, messages: Sequence[Message]) -> list[ExtractedMemory]: ...
