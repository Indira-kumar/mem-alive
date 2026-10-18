# v1 architecture decision record (draft)

Date: 2026-10-18. This records the current implementation and the release decisions still being validated; it is not a claim that v1 is ready.

## Baseline before this branch

The package has an async `Memory` facade, three stores, a shared record schema, an in-memory vector backend, and an Ollama embedding adapter. Writes accept already-extracted text; there is no conversation-to-facts pipeline. Semantic recall filters superseded records, episodic recall applies a time decay, and procedural recall combines vector and token overlap. Existing tests cover basic round trips with fake embeddings.

## Release-critical gaps found in baseline code

| Area | Current behavior | v1 requirement | Evidence |
| --- | --- | --- | --- |
| Durability | Only `InMemoryBackend` exists; the declared LanceDB extra has no implementation | Reopen the same local store and recover records | Restart round-trip eval |
| Fact safety | Vector similarity alone marks a semantic fact superseded | A similar but compatible fact must remain visible; corrections require an explicit, inspectable decision | Similar-fact and correction eval |
| Memory lifecycle | Facade has only `remember` and `recall` | Get, list, update, and delete by ID within a namespace; preserve a change trail | Lifecycle eval |
| Retrieval | Federated recall concatenates three independently limited lists | Return one ranked, globally limited result with inspectable scoring and source | Retrieval eval with cross-type negatives |
| Data integrity | In-memory backend keys records globally by ID, normalizes zero vectors without a guard, and does not validate dimensions | Namespace-safe IDs and finite, dimension-consistent vectors | Backend boundary tests |
| Ingestion | Caller must supply a single fact and memory type | Extract useful facts, episodes, and procedures from conversations using a replaceable model adapter; permit raw writes | Extraction quality eval |
| Document retrieval | No document or section model | Keep document trees separate from atomic memories; retrieve cited sections through a PageIndex adapter | Document navigation eval |
| Operations | No migration, backup, privacy, latency, or recovery contract | Document local data path, export/backup, versioning, and failure behavior | Restart and backup drill |

## Proposed boundaries

```text
Agent / application
  -> Memory API (scope, lifecycle, recall)
     -> ingestion (extract, classify, decide)
     -> atomic stores (semantic / episodic / procedural)
     -> document retrieval (PageIndex tree + cited source spans)
     -> retrieval policy (candidate generation, rerank, budget, explain)
     -> durable record and index backend
```

Atomic memory and source documents have different jobs. Facts and preferences need explicit identity, updates, provenance, and deletion. PageIndex navigates long source documents by hierarchical section; its retrieved sections should carry document and page/section citations. A document hit may support a fact, but should not silently become one. This separation permits vector, keyword, and tree retrieval to contribute candidates without making one storage system responsible for all three.

## Decisions proposed for a personal local v1

1. Keep `namespace` as the hard isolation key. Add structured `source` and timestamps to records before automatic ingestion. Do not use arbitrary metadata as an authorization boundary.
2. Make writes conservative. Similarity retrieves candidates; only an explicit update or a decision adapter that returns `replace` may supersede an existing fact. Preserve the prior version in history.
3. Use an embedded durable backend for the first personal installation. Storage transactions must cover a correction and its history event together. Do not advertise durability until a process-restart eval passes.
4. Return ranked hits with score, memory type, and provenance. The existing list-of-records API can remain a convenience wrapper during migration, but its global `top_k` must be honored.
5. Treat PageIndex as an optional document subsystem. Keep the base package usable without an API key or its heavier dependencies. Test navigation on an answerable long document and an unanswerable query; report citation correctness separately from answer quality.
6. Make the first distribution a reviewable release candidate. Tagging and publishing follow only after all acceptance gates pass and the owner reviews the PR.

## Release sequence

1. **Safe foundation:** fix destructive semantic behavior and backend invariants; add lifecycle operations and deterministic evals.
2. **Durable personal use:** implement embedded persistence, history, backup/export, and restart eval.
3. **Memory quality:** add replaceable extraction and decision adapters, live model evals, global retrieval ranking, and measurable baselines.
4. **Structured sources:** add PageIndex document ingestion and cited retrieval with an optional dependency and document eval.
5. **Release:** install into one real agent workflow, run end-to-end recovery and quality checks, document limits, then review the PR and publish.

The sequence is an acceptance map, not a promise that all stages have passed. A single day is a constraint on scope, not a reason to label an unverified build v1.

## Implemented on `codex/v1-foundations`

- Conservative semantic writes and explicit correction by ID; similar content no longer causes silent deletion.
- Namespace-safe in-memory IDs and vector boundary checks.
- SQLite persistence with schema versioning, namespace-scoped get/list/delete, transactional history, private file permissions, and an online backup.
- Scored recall with a global limit; backend candidate search filters memory type before applying its limit.
- Optional PageIndex text-PDF adapter with direct section-tree and source-page navigation, page citations, source-text verification for uncited numeric answers, and a local document eval.
- Ollama conversation extraction with user-only provenance and duplicate suppression; raw transcript text is opt-in.
- JSON CLI for separate agent processes and local document operations.
- Deterministic safety, restart, retrieval, and adapter evals plus small live Ollama quality baselines.

## Comparison with current tools

Mem0 already supports add, search, get, list, update, delete, and history; its current documentation also describes semantic, keyword, entity, and temporal retrieval signals. Its platform has graph features. We should therefore measure a specific improvement rather than claim that hybrid search or history is absent. The first useful distinction is an inspectable, conservative correction policy for personal memory, plus cited navigation of long source documents alongside atomic memories.

PageIndex's local SDK indexes text-based PDFs into a section tree and can return page citations. It does not replace atomic fact lifecycle, and local indexing does not read image-only PDFs. A v1 integration needs a live eval that checks answer correctness, citation precision, abstention on unanswerable questions, latency, and cost on the owner's actual documents.

Sources checked 2026-10-18: [Mem0 operations](https://github.com/mem0ai/mem0/blob/main/mem0/AGENTS.md), [Mem0 retrieval](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/how-it-works.mdx), [PageIndex SDK document processing](https://docs.pageindex.ai/sdk/documents), [PageIndex citations](https://docs.pageindex.ai/sdk/chat).

## Proposed release measurements

| Gate | Proposed measure | Status |
| --- | --- | --- |
| No silent fact loss | Zero compatible facts hidden in a curated similar-fact set | Deterministic policy eval passes; live set pending |
| Persistence | Add, correct, restart, search, history, and backup restore succeed | Deterministic restart eval passes |
| Retrieval | Correct type survives candidate crowding; global limit holds | Deterministic retrieval eval passes; real relevance set pending |
| Document answers | Cite the correct page for answerable queries and abstain on unsupported queries | Direct tree/page navigation passes; model chat fails across repeated runs; personal document eval pending |
| Conversation extraction | Save grounded user preferences, events, and workflows while rejecting assistant guesses and small talk | Six-case live eval passes; broader corpus pending |
| Daily use | One real agent workflow captures and recalls useful memory across sessions | CLI cross-process eval passes; deployment in owner's agent workflow pending |

Observed model settings, metrics, and failures are in [eval results](eval-results.md). Local setup and recovery are in [operations](operations.md); a proposed agent integration is in [agent workflow](agent-workflow.md).
