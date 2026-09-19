# Evals and LanceDB

This document records the design decisions behind `spec/002_evals.md`.

## Eval architecture

The eval path is deliberately separate from the memory stores:

1. `EvalRunner` seeds an isolated namespace for each case.
2. `RagAgent` recalls the requested memory type and builds a grounded prompt.
3. `ChatProvider` generates an answer. `OllamaChatProvider` is the local implementation.
4. The runner measures exact context recall, context precision, and answer-term coverage.

The built-in benchmark contains one case for every memory policy. Tests run the same cases with deterministic providers, while `python -m mem_alive.evals` uses local Ollama models. This keeps CI repeatable without replacing the live-model quality run.

Run the deterministic suite:

```bash
uv run --package mem-alive --extra lancedb --group dev pytest -q
```

Run the live local-model eval:

```bash
uv run --package mem-alive python -m mem_alive.evals \
  --embedding-model embeddinggemma \
  --chat-model qwen3:14b
```

The live command prints a JSON report and exits with status `1` if any case fails.

## LanceDB architecture

`LanceDBBackend` implements the existing `StorageBackend` contract without leaking persistence details into the stores.

- The optional dependency is imported lazily; importing `mem_alive` still works without the extra.
- The table is created on the first write because the embedding dimension is only known then.
- `namespace` and `id` form the upsert identity.
- Metadata is stored as canonical JSON, retaining arbitrary JSON-compatible dictionaries.
- Namespace filtering is pushed into LanceDB. Metadata subset matching is applied over the namespace's distance-ranked candidates to preserve the in-memory backend's exact semantics.
- Cosine distance is converted back to the similarity score expected by the stores.

The exhaustive metadata-filter path favors correctness for the first persistent release. At larger namespace sizes, metadata normalization and indexed filter pushdown should replace it; that work belongs with scale benchmarks so optimization remains evidence-driven.

LanceDB's native engine cannot open its local runtime inside some filesystem sandboxes. In that environment, run its test file with the database process permitted to access the local temporary directory.
