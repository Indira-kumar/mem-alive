# Local operations and recovery

## Storage

The default CLI database is `$MEM_ALIVE_DB`, otherwise `$XDG_DATA_HOME/mem-alive/memory.db`, otherwise `~/.local/share/mem-alive/memory.db`. The SQLite backend sets the database file to mode `0600` and creates new parent directories with mode `0700`. The CLI also sets a `077` umask. The database is not encrypted; filesystem access and existing backups remain the operator's responsibility.

Each record belongs to one namespace. The SQLite schema is at version 2. It records an add or update snapshot in the same transaction as the record change. Auditable `delete` stores a deletion snapshot. `forget` removes the active record and its local event history; it does not erase previously created backups or filesystem snapshots. SQLite's secure-delete setting is enabled for each connection, but physical erasure across storage layers is not guaranteed.

Opening a schema version 1 file creates the version 2 event table. Earlier operations cannot be reconstructed, so their history starts at the first write made by this version.

`SQLiteBackend.backup(destination)` uses SQLite's online backup API and refuses to overwrite a destination. Restore by stopping writers, preserving the current file, and starting the CLI with `--db path/to/backup.db`. Check `get`, `history`, and `recall` in the expected namespace before replacing any live file. The PageIndex document library is under a sibling `documents/` directory and is **not included** in the SQLite backup; copy that directory separately if document recovery matters.

## Models

The default memory embedding model is Ollama `embeddinggemma`; changing embedding models for an existing namespace can change vector dimensions and is rejected. Reindex existing records before switching. `llama3.2` is the default conversation extractor and local PageIndex model. The provider timeout is 60 seconds for embeddings and 90 seconds for extraction. Set `--ollama-url`, `--embedding-model`, or `ingest --model` as needed. PageIndex local mode handles text-based PDFs; scanned or image-only PDFs need OCR or PageIndex Cloud.

## Verification

From the repository root:

```bash
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra pageindex --group dev pytest -q
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra pageindex --group dev ruff check .
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra pageindex --group dev ty check
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv build --package mem-alive
```

The deterministic evals are `evals/safe_memory.py`, `evals/durable_memory.py`, `evals/retrieval.py`, and `evals/document_contract.py`. With Ollama running and the two models installed, run `evals/live_embeddings.py`, `evals/live_ingestion.py`, `evals/pageindex_live.py`, and `evals/cli_roundtrip.py` through `uv run --package mem-alive --extra pageindex --group dev python <path>`. Live evals use small synthetic examples and do not prove general quality on personal documents.

## Current limits

- Semantic corrections are explicit. Similarity never silently replaces an old fact. Conversation extraction can add a corrected preference while an older one remains until an agent calls `update` or `delete`.
- SQLite vector search scans one namespace and the PageIndex fallback may read all pages when a numeric answer lacks a native citation. Measure latency before using either on large collections.
- Federated scores combine different store policies and are not calibrated relevance probabilities across memory types.
- PageIndex fallback verifies numeric answers against source text. Uncited nonnumeric answers are rejected. A citation supports a page match; it is not proof that every inference in the answer is correct.
- The local extraction and document evals cover a few synthetic cases. Before tagging v1, run a representative private corpus, record answer accuracy, false memories, citation correctness, abstention, latency, and recovery results.
