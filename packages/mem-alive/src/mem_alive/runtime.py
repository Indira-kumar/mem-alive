"""Local defaults shared by executable entry points."""

from pathlib import Path

import httpx

from .backend.sqlite_backend import SQLiteBackend
from .core.agent_memory import AgentMemory
from .embedding.local_embedding_provider import LocalEmbeddingProvider
from .inference import JSONInferenceProvider
from .inference.provider import validate_text


def build_memory(
    db_path: str | Path,
    model: str = "qwen3:8b",
    embedding_model: str = "embeddinggemma",
    base_url: str = "http://localhost:11434",
) -> AgentMemory:
    """Use Ollama for inference/embeddings and SQLite for all persistent state."""
    # Validate before opening the database or creating owned HTTP clients.
    validate_text(model, "model")
    validate_text(embedding_model, "embedding_model")
    url = httpx.URL(base_url)
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("base_url must be an HTTP(S) URL")
    return AgentMemory(
        SQLiteBackend(db_path, embedding_model=embedding_model),
        LocalEmbeddingProvider(base_url=base_url, model=embedding_model),
        JSONInferenceProvider(model, base_url=base_url),
    )
