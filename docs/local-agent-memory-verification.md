# Verification record

Recorded 2026-11-04 (Asia/Kolkata), on the isolated Linux aarch64 VM using
CPython 3.14.4. Machine-readable command outputs are in
[evidence](evidence/local-agent-memory.json).

## Passing deterministic checks

- `uv run --locked --package mem-alive --all-extras --group dev pytest -q`: **139 passed**.
- Ruff and ty: **passed** with all extras installed.
- `python -m evals.sqlite`: **9/9**.
- `python -m evals.agent_memory`: **7/7**, explicitly scripted orchestration mode.
- `python -m evals.hybrid`: **5/5**.
- `python -m evals.mcp`: **9/9**, official SDK sessions and HTTP requests over SQLite.
- `uv build --package mem-alive`: wheel and source archive built at version **0.1.1**.
- Fresh isolated wheel installation: base imports and MCP console-script registration **passed**.
- `python -m evals.release_workflow`: invalid version rejected and selected test version **99.88.77** appeared in both distributions; no upload performed.
- `git diff --check`: **passed**.

The recorded test/lint/type commands use the same installed, locked environment
via `.venv/bin`; development instructions show the equivalent uv invocation.
Tests exercised transport and provider failures, schema migration, scoped
concurrent writes, long-turn input, deduplication after updates, and atomic failed
capture. Independent review identified and fixed startup resource creation before
validation and long-turn query-size handling.

## Live quality remains blocked

`python -m evals.agent_memory --live` exited **2**, reporting an environment
`ConnectError` for local Ollama. Settings: `qwen3:8b`, `embeddinggemma`, temperature
0, abstention threshold 0.3. This is not a passing quality result.

The VM has about 3.6 GiB available RAM, insufficient for the default 8B model.
The official ARM64 Ollama runtime (0.40.14) was downloaded into
`/home/coder/projects/.mem-alive-model-runtime` to try a smaller model locally.
Startup unexpectedly generated Ollama's default local signing-key pair. The
runtime was stopped and both generated key files were removed. The owner's VM
instructions require approval before adding credentials, so explicit approval
was requested before any further startup/model downloads. No personal credentials,
host services, or host model endpoints were used. No weights were downloaded.

After approved provisioning, run the same live eval with a documented model that
fits the VM, preserving all quality criteria and thresholds. Record real results,
settings, and variance before production release. A small seven-case lifecycle
eval is a smoke baseline, not a production benchmark or Mem0 quality-equivalence
claim; the held-out retrieval and paired coding-agent studies remain outstanding.

## Existing legacy diagnostic remains visible

`python -m evals.core --strict` exited **1**: **5/7** for the unchanged original
`Memory` API. Known failures are similarity-based overwriting of compatible
facts and a federated top-K overflow. These are not new `AgentMemory` results.
The new service covers compatible-fact retention and a global limit separately.
The older core-contract, held-out retrieval, and paired coding corpora are not
claimed as passed. Retaining the old API does not make those limitations suitable
for a v1-wide production claim.

## Promotion state

Feature branch: `feature/local-agent-memory` in `/home/coder/projects/mem-alive`.
No push, PR, tag, deployment, or publication has occurred. Package version was
not promoted to 1.0.0. The owner must approve credentials and pushing under the
VM instructions, and review/merge remain the owner's actions.
