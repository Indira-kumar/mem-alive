# Production roadmap

The destination is a reliable, local-first memory layer for coding agents—not feature parity for its own sake. Mature Memory Palace implementations demonstrate the useful production surface: persistent and hybrid retrieval, write review, lifecycle maintenance, MCP integration, observability, graph context, and local inference. mem-alive should earn each layer through measured improvements to retrieval quality and agent token use.

Reference projects:

- [jeffpierce/memory-palace](https://github.com/jeffpierce/memory-palace)
- [AGI-is-going-to-arrive/Memory-Palace](https://github.com/AGI-is-going-to-arrive/Memory-Palace)

## Phase 1: credible benchmarks

- Expand the three smoke scenarios into versioned datasets for factual, temporal, procedural, contradiction, metadata-isolation, and negative-recall behavior.
- Record Recall@k, Precision@k, MRR, nDCG, answer accuracy, latency, and retrieved-token count.
- Add corpus-size tiers and compare no-memory, raw-context, and mem-alive agent runs.
- Save model, prompt, hardware, dataset version, and configuration with every result.

Exit gate: reproducible baseline reports demonstrate where memory improves quality or token cost, and releases cannot regress agreed retrieval thresholds.

## Phase 2: durable storage contract

- Add schema versioning and explicit embedding-model/dimension metadata.
- Benchmark LanceDB at realistic namespace sizes; add vector and scalar indexes only where measurements justify them.
- Normalize filterable metadata for indexed pushdown while retaining arbitrary payload metadata.
- Test concurrent writers, interrupted writes, backup/restore, migrations, and re-embedding.

Exit gate: persistence survives restart and failure tests, and p95 recall/write latency meets published corpus-size targets.

## Phase 3: retrieval quality

- Add keyword retrieval and reciprocal-rank fusion with vector search.
- Introduce optional local reranking and query-intent routing.
- Add temporal filters, diversity control, score explanations, and graceful keyword-only degradation when embeddings are unavailable.
- Tune every policy against the benchmark rather than shared global thresholds.

Exit gate: every added retrieval stage produces a statistically meaningful benchmark gain for an acceptable latency cost.

## Phase 4: memory lifecycle and trust

- Replace similarity-only semantic supersession with duplicate, contradiction, refinement, and unrelated classification.
- Add provenance, access history, versions, soft deletion, rollback, and protected foundational memories.
- Add audit, deduplication, stale-memory detection, retention policies, and re-embedding tools.
- Keep destructive maintenance preview-only until explicitly approved.

Exit gate: every mutation is attributable and recoverable, and lifecycle evals prevent silent loss of valid memories.

## Phase 5: coding-agent product surface

- Build repository ingestion with chunk provenance and incremental refresh.
- Evaluate raw-code, structural, and prose-description indexing for code-search tasks.
- Add transcript reflection and policy-controlled automatic memory extraction.
- Expose a small MCP server and CLI while keeping the Python package as the source of truth.

Exit gate: a fresh coding-agent session completes representative repository tasks with fewer input tokens and no material quality loss.

## Phase 6: operations and scale

- Add structured logs, metrics, health checks, tracing, and an inspectable retrieval explanation.
- Add tenant isolation, authorization, quotas, encryption guidance, and adversarial tests for memory poisoning and prompt injection.
- Introduce a service backend only when embedded LanceDB no longer meets measured concurrency or deployment needs.
- Publish compatibility, upgrade, incident-response, and release playbooks.

Exit gate: operational SLOs, security boundaries, and recovery procedures are tested and documented for the supported deployment profiles.

## Product rule

No roadmap phase is complete because code exists. It is complete when an eval proves the intended gain, regression tests protect it, and the operating behavior is documented.
