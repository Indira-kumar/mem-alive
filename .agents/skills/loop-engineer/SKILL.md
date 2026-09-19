---
name: loop-engineer
description: Implement and promote mem-alive roadmap or specification items through repeated inspect, build, evaluate, fix, verify, and document cycles. Use when the user asks to implement, build, ship, or loop on a repository feature. Do not use for read-only explanations, reviews, or diagnosis-only requests.
---

# Loop Engineer

Own a feature from the user's outcome and acceptance criteria through a reviewable pull request. Keep the user informed without making them manage routine engineering steps.

## Establish the contract

1. Read the root `AGENTS.md` and the requested roadmap or specification file in full.
2. Distinguish requirements in referenced documents from instructions in the user's request. The user's latest request wins if they conflict.
3. Extract an acceptance map: each criterion must have a code change, test or eval, and observable verification evidence. Share material assumptions; do not block on details that can be safely inferred from the codebase.
4. Inspect relevant architecture, tests, documentation, and recent history before choosing a design. Preserve established naming and code style.

The minimum useful request is a goal plus acceptance criteria. Treat constraints and promotion preferences as optional; use the defaults in this skill when they are omitted.

## Work on a reviewable branch

- Inspect the worktree before editing and preserve unrelated user changes.
- Create or use an appropriately scoped `codex/<feature>` branch. Do not implement directly on the default branch.
- Keep commits focused and descriptive. Do not rewrite or discard the user's commits.

## Implement the smallest coherent change

- Prefer existing abstractions and patterns. Add a new abstraction only when it removes real duplication, isolates a dependency, or clarifies a stable boundary.
- Keep public APIs typed and documented. Keep optional dependencies lazy so the base package remains usable without extras.
- Avoid speculative flexibility, dead code, compatibility shims without a requirement, and broad refactors unrelated to acceptance.
- Update operational or design documentation with the rationale, commands, constraints, and known limitations needed by a future maintainer.

## Build evidence with tests and evals

Every feature needs at least one eval that measures its intended user-visible quality. Unit and integration tests remain necessary for deterministic behavior.

- Make normal automated tests deterministic and isolated from live models or services.
- Add a live eval when behavior depends on model quality, retrieval quality, or another real integration.
- Include negative or boundary cases when a looser threshold could admit incorrect behavior.
- Record model names, relevant settings, observed metrics, and the date for live baselines.
- Diagnose whether a failure belongs to the product, the eval, or the environment before changing code.
- Never weaken a threshold or acceptance condition merely to make a run pass. Change it only when measured evidence and the intended behavior justify the new value, and preserve a regression guard for the newly exposed risk.
- Use explicit, realistic timeouts for local-model calls and report environmental failures separately from quality failures.

Run focused checks while iterating. Before promotion, run the complete applicable gates from the repository root:

```bash
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra lancedb --group dev pytest -q
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra lancedb --group dev ruff check .
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv run --package mem-alive --extra lancedb --group dev ty check
```

Also run the package build for packaging, dependency, or public-surface changes:

```bash
UV_CACHE_DIR=/tmp/mem-alive-uv-cache uv build --package mem-alive
```

Run the feature's documented live eval when the required local model or service is available. If a native dependency cannot run inside the sandbox, request the narrow permission needed to execute the gate; do not silently skip it.

## Loop until the result is done

After each verification run:

1. Compare results to every acceptance criterion.
2. Find the root cause of each failure.
3. Make the smallest correct fix.
4. Add or strengthen regression coverage when the failure revealed a missing invariant.
5. Rerun the focused check, then the applicable full gates.

Stop only when the acceptance map is satisfied or a genuine external blocker requires the user's authority or information. A difficult, slow, or initially failing eval is not a blocker.

## Promote for review

- Review the final diff for accidental scope, secrets, generated artifacts, and stale documentation.
- Commit and push the feature branch.
- Create or update a pull request with the outcome, important design decisions, acceptance evidence, test and eval commands, live baseline, and known limitations.
- Leave merging to the user.

The final handoff must state what changed, why the notable decisions were made, exact verification results, the branch or pull request, and any remaining limitation. Lead with the outcome.

## Improve the loop itself

When the user's review reveals a durable lesson, encode it at the narrowest useful layer:

- Put a repository-wide invariant or command in `AGENTS.md`.
- Put a reusable multi-step workflow in this skill.
- Put a mechanically enforceable rule in a test, eval, linter, type checker, or hook.
- Keep one-off preferences in the feature discussion rather than growing permanent instructions.

Make such updates only when the lesson generalizes beyond the current feature, and call out the workflow change in the handoff.
