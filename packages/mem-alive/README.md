# mem-alive

[![PyPI](https://img.shields.io/pypi/v/mem-alive)](https://pypi.org/project/mem-alive/)

A memory layer for AI agents. Semantic, episodic, and procedural memory, built as a Python package.

## Why

Coding agents re-read the whole codebase every time they start a task. That burns tokens and adds round trips between the cloud model and the local harness. This library exists to fix that by giving agents actual memory instead of a blank slate every session.

That said, it's not built just for coding agents. It's a general memory layer that works with any agentic setup or a plain RAG app. The coding-agent problem is the flagship test case, not something baked into the core.

## Architecture

![architecture diagram](https://raw.githubusercontent.com/Indira-kumar/mem-alive/main/packages/mem-alive/artifacts/v0.1-architecture-diagram.png)

One `Memory` client fronts three stores, each backed by the same pluggable `StorageBackend` and `EmbeddingProvider`.

## Three kinds of memory

- **Semantic** - durable facts, no recency weighting. New facts can supersede old, similar ones.
- **Episodic** - specific past events, timestamped, recency-weighted (exponential half-life decay), never merged.
- **Procedural** - skills and workflows, retrieved by hybrid search (embedding similarity + keyword overlap).

Each type has its own store with its own retrieval logic, but all three share one schema and one `memory_type` tag, so a single federated `recall()` on the `Memory` client can query across all of them at once.

## Scoping

- `namespace` - hard partition, never crossed. Means whatever the caller wants (agent, repo, tenant).
- `metadata` - flexible filters within a namespace (session id, tags, etc).

## Contradictions

Semantic writes check for contradictions on every `remember()`: embed the new fact, search for similar existing facts in the same namespace, and if similarity crosses a threshold, mark the old fact as superseded. v0.1 uses a similarity threshold for this. A smarter LLM-arbiter version (duplicate vs contradiction vs refinement) is a future upgrade, not required for the first release.

## Storage

The backend stays dumb: vector search, metadata filters, CRUD, nothing else. Recency decay, hybrid scoring, and contradiction logic all live above it, in the store layer, so any backend stays swappable.

v0.1 ships with `InMemoryBackend`, good for development and testing. `LanceDBBackend` is the persistent, embedded option and is installed separately with `pip install mem-alive[lancedb]`.

```python
from mem_alive import LanceDBBackend

backend = LanceDBBackend("./data/memories")
```

LanceDB fixes the embedding dimension when the table is first created. Changing embedding models to one with a different dimension requires a new table or a deliberate migration.

## Evals

The package includes an agentic RAG eval harness with semantic, episodic, and procedural scenarios. It uses Ollama for both local embeddings and answer generation:

```bash
python -m mem_alive.evals \
  --embedding-model embeddinggemma \
  --chat-model qwen3:14b \
  --timeout 300
```

Each case measures retrieved-context recall, retrieved-context precision, and required answer-term coverage. The command exits non-zero when any case misses its quality thresholds, so it can also serve as a release gate.

The `Memory` facade uses a `0.6` cosine-similarity threshold, calibrated against the live local-model evals. Applications can choose a stricter operating point with `Memory(..., recall_threshold=0.8)` and should validate it against their own model and corpus.

## Status

v0.1.1. Core is done: all three stores, in-memory and LanceDB backends, a local embedding provider (Ollama), the federated `Memory` client, and a local-model eval harness. All I/O paths are async and covered by unit, integration, and eval tests. MIT licensed.

Still open: a coding-agent app layer, larger benchmark datasets, latency baselines, and token-savings measurements.
