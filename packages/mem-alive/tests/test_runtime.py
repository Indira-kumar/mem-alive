"""Reject bad startup configuration before creating persistent resources."""

import pytest
from mem_alive.runtime import build_memory


@pytest.mark.parametrize(
    "options", [{"model": ""}, {"embedding_model": " "}, {"base_url": "file:///tmp/model"}]
)
def test_invalid_configuration_creates_no_database(tmp_path, options):
    path = tmp_path / "memory.sqlite3"
    with pytest.raises(ValueError):
        build_memory(path, **options)
    assert not path.exists()
