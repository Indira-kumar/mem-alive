"""Typed model boundary for contextual extraction, entities, and rolling summaries."""

from typing import Protocol

MAX_ITEMS = 100
MAX_TEXT_LENGTH = 16_000


def validate_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_LENGTH:
        raise ValueError(f"{label} must be nonblank text of at most {MAX_TEXT_LENGTH} characters")
    return value


class InferenceProvider(Protocol):
    """A replaceable model boundary; implementations never mutate storage."""

    async def extract(
        self, messages: list[dict[str, str]], *, context: dict | None = None
    ) -> list[str]: ...

    async def extract_entities(self, texts: list[str]) -> list[list[str]]: ...

    async def summarize(self, previous_summary: str, messages: list[dict[str, str]]) -> str: ...
