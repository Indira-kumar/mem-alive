# Local agent memory: design and acceptance record

## Contract and sources

The owner requested the architecture in Hugging Face's video, a usable local MCP,
clean code, tests and feature evals, a reviewable feature branch, and preparation
for v1.0.0. The owner supplied the [timestamped transcript](https://www.usetranscribe.io/yt/aYfZN8t6AQs/agent-memory-explained),
which was read on 2026-11-04. It supersedes the initial interpretation based on
Mem0's older two-stage reconciliation paper. No ADD/UPDATE/DELETE/NOOP inference
pipeline remains in the new service: capture is additive with exact deduplication.
Explicit management updates/deletes remain supported.

The transcript establishes contextual extraction; main and entity vector stores;
SQLite history and last-ten messages; a semantic pool of max(4*K,60); BM25 and
entity reranking restricted to that pool; and division by 2.5. It does not reveal
the on-screen entity formula. Upstream [main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)
and [scoring.py](https://github.com/mem0ai/mem0/blob/main/mem0/utils/scoring.py)
were inspected to resolve the formula, rather than inventing an exactness claim.
Upstream is a moving reference, not a pinned compatibility contract.

## Boundaries and rationale

`AgentMemory` orchestrates contextual extraction and retrieval. The inference
protocol has three model operations: fact extraction, entity extraction, and
rolling summary generation. Ollama and OpenAI-compatible HTTP transports share
strict structured-output validation. No heuristic extraction fallback disguises
provider failures. Embeddings are injected; the local runtime defaults to Ollama.

`SQLiteBackend` owns main-memory vectors, a separate entity-vector/link table,
context and last-ten messages, mutation history, schema migration, exact hash
uniqueness, embedding identity/dimension checks, and transaction boundaries. Both
vector collections use exact cosine search. Keeping logical collections in one
embedded database allows atomic capture, avoiding partially updated external
vector stores. This differs physically from Mem0's configurable external stores;
no approximate-nearest-neighbor throughput is claimed.

All model calls finish before committing capture. A per-instance async lock
serializes captures; per-namespace durable revision tokens reject cross-instance
stale plans, including phantom inserts. Explicit updates compare the previous
record before writing. Conflicts surface as retryable errors, never silent loss.
Concurrent callers should retry an entire failed capture with fresh context.
Cancellation while a SQLite worker is committing may arrive after an atomic
commit; callers should inspect state before assuming cancellation meant rollback.

Only the last 16,000 characters of flattened new messages are embedded for
ingestion-context retrieval; the extractor receives the complete original turns.

Recent messages and summaries are namespace scoped. Choose a namespace matching
the identity whose conversational context should be shared; metadata narrows
retrieval but does not create an independent conversation buffer. Different users
must not share a namespace unless that context sharing is intentional.

The reranker exposes semantic, keyword, entity, and final scores. English tokens
are dictionary-lemmatized; keyword tokens persist with the memory. BM25 corpus
statistics are computed over the scoped semantic pool. Threshold and top K apply
to final scores per the adopted video flow. The original `Memory` API remains
unchanged and is not the release-candidate application service.

## Acceptance map

| Requirement | Implementation | Evidence |
|---|---|---|
| Local restart durability | SQLite main/entity/context/history | storage tests, `evals.sqlite`, `evals.mcp` |
| Contextual fact extraction | summary, recent/relevant memories, last ten turns and dates | inference/service tests, `evals.agent_memory` |
| Exact deduplication | SHA-256 text hash with scoped unique index | storage/service tests and lifecycle eval |
| Separate linked entity store | persisted vectors/links, atomic replacement and deletion | storage tests, hybrid eval |
| Candidate pool and hybrid score | max(4*K,60), BM25, bounded entity boost, /2.5 | retrieval tests, `evals.hybrid` |
| MCP coding-agent usage | seven official SDK stdio tools, fixed namespace | real SDK ClientSession protocol test and restart eval |
| Other application surfaces | async Python API, optional ASGI HTTP, compatible inference endpoints | provider transport tests, HTTP protocol tests/eval |
| No partial failed capture | prevalidation plus single transaction and revision checks | embedding/entity failure and concurrent storage tests |
| Reviewability | feature/local-agent-memory, design/run instructions, local commit | branch and final diff review |
| Real model quality | actual Ollama extraction/entities/summary/embedding | `evals.agent_memory --live`; blocked without local models |

## Release boundaries

This work does not claim Mem0 SDK, API, backend-provider, hosted-product, or every
framework adapter parity. MCP clients and Python/HTTP consumers can integrate the
same service; client configuration and automatic turn hooks remain host-owned.
The existing advanced evidence/procedure and paired coding-agent corpora are not
silently reclassified as passing. No paired agent benchmark or live retrieval
baseline has been fabricated.

The package version remains 0.1.1 until maintainer-selected release promotion.
The existing release workflow selects the requested version during publishing.
Do not release v1 solely from deterministic scripted evals. A running local
model server/weights and a passing recorded live baseline are still required.
No host-machine model endpoint, browser session, or isolation setting was used.

Deletion retains audit history and may leave source facts in prior context and
summaries. Full data erasure, encryption, remote authentication, expiry enforcement,
and large-scale ANN indexes are not implemented. Caller metadata can record owner,
agent, run, session, and expiration information but expiry is not silently promised.

## Verification

Exact commands, counts, and the live-environment outcome are recorded in
[verification](local-agent-memory-verification.md). No push, publish, deployment,
or pull request creation is authorized by this branch's local implementation.
