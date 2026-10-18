# mem-alive: context for the evaluation workstream

This document is a self-contained handoff for a separate thread designing and building evals. It records the owner's stated goals and decisions, the proposed architecture, and the current repository state as of 2026-10-18. **The owner's requests and acceptance criteria remain authoritative.** Sections marked *proposed* describe design direction, not settled product requirements. Evals for future behavior may initially fail against the current implementation; report that honestly rather than weakening them to fit today's code.

## Product goal

Build a reusable Python memory layer for agentic systems. The first application is local coding agents such as Codex and Claude Code, but the core should also work for non-coding agents such as Instinct or Hermes through adapters or an SDK. A user should not have to repeat valuable context in a later session, and the agent's task quality should stay the same or improve. The memory layer should supply context automatically when relevant, and supply nothing when it lacks reliable information.

The owner values clean boundaries, small purposeful code, evidence-backed decisions, and a repository others can learn from. The initial release should be installable and useful to anyone running a local coding agent. There is no need to compress the architecture roadmap to a single day; work can be developed and evaluated one part at a time.

### Settled decisions from the design discussion

- Inputs include conversation messages/transcripts and tool results. Capture must be independent of the memory extraction process.
- The knowledge chain is **source data → selected evidence → derived memory**. A memory is not its own evidence. Every consequential memory should be traceable to source evidence.
- Memory types are **semantic/durable**, **episodic**, and **procedural**. Entities are an orthogonal way to identify subjects and relationships, not a fourth memory type.
- Episodic memory records meaningful past events and outcomes, not every conversational turn. Keep the event distinct rather than silently rewriting its history.
- One **verified** procedural success creates a candidate; repetition across independent attempts can promote it to a procedure. Agent self-report alone does not verify success. Observed outcomes or user confirmation can verify it.
- Store the information/context, source, time, confidence, and access count/strength. Access count is a usefulness signal, **not evidence of truth**.
- The ingestion pipeline decides whether to create, refine, supersede, or dispute memory. It also infers scope. A general technology or user practice can follow the user across projects; project-specific knowledge stays scoped to that project. A conservative narrow scope is appropriate when the evidence does not establish generality.
- Support some version history. A later fact should not erase the evidence for an older version. Relevant unresolved disputes may be surfaced with qualification; irrelevant disputes should not interrupt the agent.
- Only insufficiently confident memories need review. Strongly supported memories can be handled automatically. Review should not become a routine interruption of the user's work.
- Retrieval and ranking must be designed deliberately. Return nothing when no reliable, relevant memory exists.
- Ingestion should run in parallel with agent work. Retrieval should be as fast as possible. Literally zero lookup latency is impossible when injecting context before a prompt; the proposed interpretation is no model/extraction work on that critical path, a measured deadline, and graceful empty output on timeout.
- Durable local storage is desired; cloud persistence may be added later. The core storage/retrieval model is independent of Codex, Claude, MCP, CLI, or any one delivery surface.
- The important coding signal is often corrective feedback: “You did Y; I wanted X.” A correction can establish a preference or constraint immediately, but a proposed workflow still needs outcome verification before being called a proven procedure.

## Architectural boundary

The intended dependency direction is:

```text
Codex / Claude / other host adapters
                ↓
Coding domain layer (only for coding use cases)
                ↓
General mem-alive core
```

**Core:** generic observation and evidence records; identity, entities, scope, versioned semantic/episodic/procedural memories; generic correction and outcome signals; verification; reconciliation; retrieval, ranking, abstention; storage; provider interfaces. Core code must not depend on repository, Git, tests, Codex, or Claude concepts.

**Coding layer:** interprets coding-specific signals such as user corrections, repo decisions, commands, paths, libraries, tests, diffs, failures, and successful workflows. It maps these into generic observations/evidence/outcomes and provides domain-specific scope and retrieval hints. It does not create a second memory database or ranking engine.

**Host adapters:** translate host lifecycle events into observations and deliver selected context. They should be thin. Codex and Claude Code both provide prompt and tool lifecycle hooks, but host coverage and behavior must be tested separately. An explicit CLI/MCP/SDK recall path may complement automatic delivery; MCP tool availability alone does not guarantee automatic recall. Relevant primary docs: [Codex hooks](https://learn.chatgpt.com/docs/hooks) and [Claude Code hooks](https://code.claude.com/docs/en/hooks).

### Proposed core API concepts (not finalized signatures)

- `observe(event)`: durably receive an event without waiting for model extraction.
- `recall(request)`: return ranked, evidence-linked memory hits under a time and token budget, possibly empty.
- `feedback(memory_id, signal)`: accept correction, confirmation, failure, or other outcome evidence.
- `inspect(memory_id)`: explain current state, evidence, versions, and disputes.
- `forget(id_or_scope)`: remove data and update derived indexes.

All identifiers should be stable enough to correlate events, evidence, memories, versions, agent sessions, users, and scopes. Extraction should be idempotent under event replay.

## Proposed processing model

### Ingestion (background)

1. A host adapter captures a user message, tool result, or completed turn into a durable local event inbox. Do not run an LLM in the hook's critical path.
2. A worker selects atomic evidence with links to source event IDs and locations. It separates user assertions, tool observations, agent claims, and explicit user confirmation.
3. A classifier proposes memory type, subject/entity, scope, temporal validity, and whether an outcome was observed or confirmed.
4. Reconciliation checks existing memories for duplicate, refinement, contradiction, or genuinely new knowledge. Semantic similarity alone cannot decide contradiction.
5. The core updates the appropriate versioned state, sends uncertain items to a review queue, and refreshes derived search indexes.

The durable inbox matters: background hooks can be interrupted at session teardown. A tiny durable event write may have a small cost, but extraction must not hold up the coding agent.

### Three memory behaviors

| Type | Create/update | Retrieve |
| --- | --- | --- |
| Semantic/durable | Current facts, preferences, constraints, and decisions. Refine, supersede, or dispute with evidence and versions. | Apply scope and current validity; do not mechanically decay a sound fact just because it is old. |
| Episodic | Meaningful event, situation, action, and outcome. Preserve its event identity; append later interpretation as linked evidence if needed. | Match a similar situation; recency can help but must not erase an older relevant lesson. |
| Procedural | Conditional workflow: trigger, preconditions, steps, and expected outcome. One verified success makes a candidate; independent repetition promotes it. Failures can narrow or demote it. | Match the task and preconditions; distinguish candidate from established procedure. |

### Retrieval (prompt-critical path)

1. Receive query/prompt plus user, agent, workspace/project, session, optional type, time, and output budget.
2. Search eligible scopes using complementary candidate generators: entity/exact lookup, keyword search, and semantic similarity.
3. Merge and deduplicate candidates. Rank **across all memory types** using relevance, reliability/evidence, scope applicability, validity, and type-specific factors. Do not directly average incomparable raw BM25 and vector scores.
4. Exclude weak or irrelevant memories. Include a dispute only when directly relevant and clearly qualified.
5. Return a small factual context pack with memory IDs/source references, or return nothing. Stored text is untrusted data, not a new instruction authority.

The suggested local storage design is SQLite as the authoritative store for events, evidence, entities, versions, and memory records; SQLite FTS5 for keyword search; a rebuildable vector index behind a backend interface. Exact vector search is acceptable initially if the measured local scale fits the latency budget. ANN and cloud storage are optimization/deployment choices, not core semantics. [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html).

## Worked correction example

Session A: An agent runs mock tests. The user says, “For this repository, run the integration test first; the mocks missed this failure.” The coding layer identifies rejected approach, preferred approach, reason, and repository scope. The core records the message as source, extracts evidence, and creates/updates the scoped preference. If the integration test subsequently passes and the outcome is observable, the core may create a procedural candidate. Independent repeated verified successes may promote it.

Session B: The user asks for related work without repeating the test instruction. Automatic retrieval returns the applicable preference, and possibly the verified procedure, before the agent decides how to test. The agent should use it without treating an unverified candidate as a proven workflow.

The same **generic** correction signal should work in a non-coding domain. The coding package detects coding-specific meaning; core state transitions remain unchanged.

## Evaluation workstream

Design evals to test behavior, not to mirror the implementation. Preserve the distinction between a **deterministic core invariant**, a **component quality eval**, and a **live agent outcome eval**. The repository requires at least one relevant eval for each feature. Record model, provider, dataset, configuration, and repeated-run variance for model-dependent results.

### A. Core invariant evals

Use small fixtures with source events, gold evidence, expected memory states, and expected retrieval behavior:

1. **Provenance:** every promoted memory resolves to its supporting evidence and source event.
2. **Replay:** ingesting the same event twice does not duplicate evidence or memory.
3. **Correction:** “prefer X over Y” updates the correct scoped preference without making a separate unrelated fact disappear.
4. **Contradiction vs similarity:** semantically similar but compatible facts coexist; contradictory facts are reconciled or disputed according to evidence.
5. **Versioning:** a superseded value is excluded from normal recall but remains inspectable with its history.
6. **Scope:** a user-wide practice follows the same user across projects; a project-specific exception does not leak into another project or user.
7. **Procedure verification:** self-report does not verify; one observed/confirmed success makes a candidate; an independent repeat promotes; a failed repeat is not counted as success.
8. **Episode integrity:** a meaningful episode and outcome remain retrievable without being mistaken for a timeless rule.
9. **Review and abstention:** insufficiently supported claims enter review and are withheld from ordinary automatic context; no reliable candidate yields an empty result.
10. **Deletion/rebuild:** removing source or memory data updates visibility and rebuilt indexes consistently.

### B. Retrieval quality evals

Use held-out queries whose wording differs from the source. Label required memories, permitted optional memories, forbidden stale/wrong-scope memories, and cases where empty output is correct. Include exact entity names/paths, paraphrases, multi-entity queries, temporal changes, and irrelevant near-matches. Measure recall of required context, precision of injected context, wrong-memory rate, abstention quality, context size, and latency distribution. Report per-type and per-scope results, not only one aggregate score.

### C. End-to-end coding evals

Create paired fresh-session tasks. Session A establishes context through real messages and tool outcomes; session B receives a related task **without repeating the missing context**. Run the same task with memory disabled and enabled, holding model, tools, repository state, and task text as constant as practical. Scenarios should include:

- user correction of a failed approach;
- repository-specific decision and conflicting user-wide practice;
- technology or general practice that should transfer across projects;
- a changed decision that supersedes an old one;
- a meaningful earlier failure or episode;
- one verified procedural success versus repeated successes;
- misleading, irrelevant, or insufficiently supported memories that should not be injected.

Judge the actual task result (tests or task-specific rubric), whether the user had to restate context, whether the agent applied the correct scoped memory, and whether memory caused a regression. A memory-on agent citing a fact but producing a worse result is **not** a success.

### D. Integration and operating evals

For each host adapter, test event capture coverage, first-session ingestion, automatic delivery in the next session, disabled/unavailable hook behavior, crash/restart recovery, duplicate events, secret/path filtering, and retrieval timeout. Measure prompt-added p50/p95/p99 time and background ingestion lag. Codex documents that tool hooks do not cover every path and its transcript format is not a stable hook API, so explicitly measure which supported events are captured rather than assuming complete transcript fidelity. [Codex hook coverage](https://learn.chatgpt.com/docs/hooks).

### Eval design safeguards

- Keep training/fixture source events separate from held-out session-B tasks. Do not put the desired answer verbatim in the prompt.
- Include negative examples and scope collisions; an eval containing only relevant memories will overstate quality.
- Assert outcomes and evidence, not a particular function call order, score formula, database layout, or prompt wording.
- Report failures by stage: capture, extraction, reconciliation, index, candidate generation, ranking, delivery, or agent use.
- Keep deterministic fixtures fast and runnable without paid model calls. Add recorded/live model evals separately with explicit provider configuration.
- Do not invent numerical pass thresholds before obtaining a baseline and understanding variance. Propose thresholds with evidence and owner review.

## Current repository state and known gaps

The package is under `packages/mem-alive/` and is currently version `0.1.1`. It has an async `Memory` client over separate semantic, episodic, and procedural stores; an in-memory vector backend; an embedding-provider interface and local Ollama provider; and unit tests for the existing components. The README documents the current v0.1 design. Inspect the live checkout before changing it, because other threads may be working concurrently.

- `src/mem_alive/schema/memory_schema.py`: one simple `Memory` dataclass with content, vector, metadata, type, namespace, superseded flag, and timestamps. It lacks source/evidence/version/verification state.
- `src/mem_alive/core/memory.py`: `remember()` delegates by type; federated `recall()` gathers each store and concatenates results without a cross-type rank.
- `src/mem_alive/store/semantic_store.py`: supersedes the closest memory when cosine similarity crosses a threshold; similarity does not establish contradiction.
- `src/mem_alive/store/episodic_store.py`: applies exponential age decay with a fixed default half-life.
- `src/mem_alive/store/procedural_store.py`: blends vector similarity with simple token overlap; it does not represent outcome verification or candidate/promotion state.
- `src/mem_alive/backend/in_memory_backend.py`: exact cosine search in memory; no persistence. `pyproject.toml` lists LanceDB as an optional extra, but there is no LanceDB backend in the checkout inspected for this document.
- There is currently no automatic event capture, evidence extraction, review queue, SQLite source of record, unified rank, coding domain package, Codex/Claude adapter, or dedicated feature eval harness in the existing source examined here.

The current code is a starting point, not the architectural source of truth. Evals should protect desired behavior and expose gaps so implementation can improve without overfitting to today's methods.

## Roadmap order and release principle

1. Define core contracts and build the first eval corpus, including a non-coding example.
2. Add durable source/evidence/version storage and replay-safe event capture.
3. Add the generic ingestion/reconciliation state machine, including corrections and verification.
4. Add scoped, unified, abstaining retrieval with a measured deadline.
5. Publish a stable generic Python API and validate it with a non-coding adapter.
6. Add the coding domain package and its correction/outcome evals.
7. Add Codex and Claude adapters and fresh-session memory-on/off agent evals.
8. Harden install, privacy, migration, repair, diagnostics, and documentation.

For each implemented feature, follow `AGENTS.md` and the repository engineering loop: a focused feature branch, at least one feature eval, relevant tests/lint/type checks, documented rationale, and a reviewable PR. Do not merge on the owner's behalf.

## Open engineering choices, not settled product requirements

The exact extractor/embedding providers, vector index implementation, ranking formula, numeric latency and quality thresholds, cloud synchronization design, retention defaults, and precise Codex/Claude packaging should be chosen through focused experiments and documented decision records. Do not let these choices change the core product boundaries or the settled behaviors above. Integration with Instinct or Hermes is a portability target, not an assertion that their current extension APIs have been validated here.
