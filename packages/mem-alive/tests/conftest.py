import pytest
from mem_alive.backend.in_memory_backend import InMemoryBackend

from .fakes import FakeEmbeddingProvider


@pytest.fixture
def backend():
    return InMemoryBackend()


@pytest.fixture
def make_embedding_provider():
    """Factory fixture: pass the vocabulary a test needs, get a FakeEmbeddingProvider back."""
    return FakeEmbeddingProvider
