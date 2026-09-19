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
  --chat-model qwen3:14b \
  --timeout 300
```

The live command prints a JSON report and exits with status `1` if any case fails.

### Live baseline: 2026-09-19

The first live loop used `embeddinggemma` for retrieval and `glm-4.7-flash:latest` for answer generation. It identified and drove fixes for an embedding request timeout, an over-strict `0.8` recall threshold, and an episodic question whose required timestamp was not requested by the question.

After those fixes, all four cases passed:

| Case | Context recall | Context precision | Answer terms |
| --- | ---: | ---: | ---: |
| Semantic production database | 1.0 | 1.0 | 1.0 |
| Episodic incident recall | 1.0 | 1.0 | 1.0 |
| Procedural key rotation | 1.0 | 1.0 | 1.0 |
| Semantic unanswerable query | 1.0 | 1.0 | 1.0 |

The unanswerable case is the guardrail for the lower threshold: its closest stored memory scored `0.317`, remained below the calibrated `0.6` default, and GLM answered that it did not know.

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
