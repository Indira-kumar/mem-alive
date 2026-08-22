# mem-alive

A memory layer for AI agents. Built as a Python package, published on PyPI.

## Why

Coding agents re-read the whole codebase every time they start a task. That burns tokens and adds round trips between the cloud model and the local harness. This library exists to fix that by giving agents actual memory instead of a blank slate every session.

That said, I'm not building it just for coding agents. It's a general memory layer that works with any agentic setup or a plain RAG app. The coding-agent problem is the flagship test case, not something baked into the core.

I'm also building a local eval setup (7800 XT, 14B models) to benchmark the library and track progress across versions I publish.

## Three kinds of memory

- **Semantic** - durable facts, no recency weighting. Timestamps exist only for versioning, so new facts can supersede old ones.
- **Episodic** - specific past events, timestamped, recency matters, never merged.
- **Procedural** - skills and workflows, stored as markdown (frontmatter + body), retrieved by hybrid search (embedding + keyword/trigger match).

Each type gets its own schema since they behave differently. Every record still carries a `memory_type` tag so results can be merged across types in a single federated `recall()` later.

## Scoping

- `namespace` - hard partition, never crossed. Means whatever the caller wants (agent, repo, tenant).
- `metadata` - flexible filters within a namespace (session id, tags, etc).

## Contradictions and consolidation

Semantic writes actively check for contradictions: embed, search for similar facts, and if similarity crosses a threshold, an LLM arbiter decides duplicate / contradiction / refinement / unrelated.

Episodic memory can consolidate into semantic memory, but it's opt-in. The library exposes a `consolidate()` method, not its own background thread. Trigger it manually, every N writes, or from whatever scheduler your app already uses.

## Storage

The backend stays dumb (vector search, metadata filters, CRUD). Recency decay, hybrid scoring, and contradiction logic live above it, in Python, so any backend stays swappable.

Procedural content lives in the same backend as its embedding, not on local disk, so multi-pod deployments don't need their own sync system.

Backends are optional extras (`pip install mem-alive[lancedb]`), core has zero hard dependency on any vector DB. Default: LanceDB, embedded and no server required.

## Still open

Public API surface, how the coding-agent use case sits on top without leaking into core, eval/benchmark design, package layout, and how benchmark results get tracked across versions.

## Status

Pre-code. Architecture first.

claude --resume 900f7f37-f853-4412-9bcd-b67985addf94
