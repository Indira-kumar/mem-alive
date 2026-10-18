# Memory evaluation suite

These evals are the executable starting point for the [product contract](../docs/evals-context-handoff.md). They test observable behavior, not a database layout or score formula. The suite is intentionally split by cost and capability.

| Layer | Corpus | Runner | Current status |
| --- | --- | --- | --- |
| Current-client core probes | Seven deterministic scenarios in `core.py` | `python -m evals.core` | Runs against `Memory` today. Five pass; two expose product gaps. |
| Future core contracts | Eleven event/evidence/state scenarios in `core_contracts.json` | `python -m evals.contracts RESULTS.json` | Scorer is ready; ingestion, evidence, inspect, and forget APIs do not exist yet. |
| Retrieval quality | Seven held-out queries in `retrieval_cases.json` | `python -m evals.retrieval RESULTS.json` | Scorer is ready; a scoped retrieval adapter is needed. |
| Coding agent outcome | Eight paired fresh-session scenarios in `coding_cases.json` | `python -m evals.coding RESULTS.json` | Scorer is ready; coding and host adapters plus task oracles are needed. |

The grader mutation tests run without models, services, or network:

```bash
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --group dev pytest -q evals
```

Run the current-client diagnostic baseline:

```bash
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --group dev python -m evals.core
```

The command exits zero when it completes and prints a result for every case. Add `--strict` to make any failed product probe exit nonzero. A diagnostic run must not be described as a passing quality gate merely because its process exits zero.

## What each layer checks

`core.py` uses fixed nonzero vectors to isolate memory-store behavior from embedding quality. Each case has a separate backend. Exact content identity makes missing, extra, duplicate, and stale context visible. The probes cover namespace and metadata isolation, semantic version visibility, compatible facts, distinct episodes, abstention, and a global `top_k` request.

The core contracts describe source events, queries, expected final state, and occasional intermediate states, including a non-coding travel preference. An adapter should observe the events, process them, query the memory layer, and normalize its output. In particular, the procedure case checks that an agent claim alone does not promote a procedure, one observed success creates a candidate, and a separate observed success can establish it. The scorer follows each memory's evidence IDs to source event IDs; a memory claiming provenance without an evidence record fails.

The retrieval corpus has exact names and paths, paraphrases, a two-entity query, a changed value, a user/project collision, an older relevant episode, and a near-match where the correct answer is empty. `required` memories must appear; `optional` memories may appear; every other hit counts as wrong. Reports expose recall, context precision, wrong-memory count, context tokens, latency, type, and scope. No numeric latency or quality threshold is invented before a live baseline.

The coding corpus keeps session-A source events separate from session-B task text. Each case requires a memory-on and memory-off run with the same model, tools, checkout state, and session-B prompt. The scorer rejects pairs whose `condition` values differ. The adapter must grade the *actual task artifact* using the case's oracle; an agent statement that it succeeded is not an oracle. The scorer requires task success, correct delivered context, no extra context, and no user restatement. It also reports whether memory-on regressed relative to memory-off. Repeated pairs can use distinct `trial_id` values, and the report shows per-case success rates. The eight scenarios cover correction, scope exception, cross-project general practice, changed decision, earlier failure, candidate and established procedures, and abstention.

## Adapter result format

Adapters own product integration. They translate implementation IDs into the fixture's semantic keys and emit a JSON object with a `runs` array. Run each fixture in an isolated state. A model-dependent adapter should include provider, model, version, generation settings, date, run seed, and oracle logs alongside the scored fields. Keep agent-visible input separate from gold labels and scoring code.

Core contract run example:

```json
{
  "runs": [{
    "case_id": "provenance-chain",
    "final": {
      "recall": {"q1": ["billing-db"]},
      "records": {"billing-db": {"type": "semantic", "state": "active", "evidence_ids": ["ev1"]}},
      "evidence": {"ev1": {"source_event_id": "e1"}}
    },
    "after": {}
  }]
}
```

For a case with `expect_after`, add a normalized state under each named event ID in `after`. The core scorer checks record identity, type, state, recall, review/inspect lists where specified, source provenance, and counts derived from the supplied records and evidence. It does not rely on an adapter's self-reported count.

Retrieval run example:

```json
{"runs": [{"case_id": "exact-entity-and-path", "hit_ids": ["target"], "context_tokens": 27, "latency_ms": 3.2}]}
```

Coding run example (one pair; the full file needs one pair per case):

```json
{"runs": [
  {"case_id": "correction-after-failed-approach", "trial_id": "1", "mode": "off", "task_passed": false, "delivered_context": [], "condition": {"model": "provider/model-version", "tools_hash": "abc", "repo_state_hash": "def", "task": "Fix the callback regression in the same repository."}},
  {"case_id": "correction-after-failed-approach", "trial_id": "1", "mode": "on", "task_passed": true, "delivered_context": [{"id": "integration-first", "status": "active"}], "user_restatements": 0, "condition": {"model": "provider/model-version", "tools_hash": "abc", "repo_state_hash": "def", "task": "Fix the callback regression in the same repository."}}
]}
```

`task_passed` must come from the independently executed task oracle. Store the command, exit status, workspace snapshot, and agent trace with the run so reviewers can audit it. A memory-on run that injects the expected fact but makes the task worse fails.

## Baseline and next integration steps

On **2026-10-18**, the deterministic current-client baseline was **5/7** passing. `similar-compatible-facts` missed one compatible fact because similarity alone superseded it. `federated-top-k` returned three contexts for `top_k=1`. These failures are retained as requirements for future work. The fixture and grader tests passed in **0.15 s** on this checkout. Fixed vectors do not measure real embedding retrieval quality.

The eleven core contracts, seven retrieval cases, and eight coding cases are *not* product-pass claims. They are scored corpus contracts awaiting adapters and, for coding, independently runnable task oracles. Implement the generic event/evidence API first, then map its output into the core contract format. Add a retrieval adapter and run repeated local-model baselines before proposing thresholds. Add coding and host adapters before running the paired agent study. Report failures by capture, extraction, reconciliation, indexing, candidate generation, ranking, delivery, or agent use. Keep observed model/configuration and run variance with every live baseline.
