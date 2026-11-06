# Draft PR: Add local persistent agent memory and MCP tools

Coding agents can now save and retrieve memory across sessions using namespace-bound
MCP tools over a local SQLite database. The async Python API and optional HTTP
adapter share contextual ADD-only extraction, exact-text deduplication, persistent
entity links, and semantic/BM25/entity ranking matching the owner-adopted video flow.

SQLite atomically commits facts, context, entity links, and revision history.
Namespace revisions reject stale concurrent captures. Model calls use strict JSON
validation; failed extraction, embeddings, or entity processing leave no partial
capture. English dictionary lemmatization and exact cosine scans keep the runtime
local without requiring a vector server.

Validation: 139 tests; lint/types; storage 9/9, lifecycle 7/7, hybrid 5/5, MCP/HTTP
9/9 deterministic evals; package build, installed-wheel smoke, and release-version
artifact eval all pass. Live model quality remains unverified: Ollama provisioning
needs owner approval for its generated local signing key. The legacy Memory API
still has its documented 5/7 diagnostic baseline. This draft does not assert full
Mem0 SDK/framework/provider compatibility or production readiness for v1.0.0.

See docs/local-agent-memory.md and docs/local-agent-memory-verification.md for
architecture choices, exact evidence, integration instructions, and release gaps.
