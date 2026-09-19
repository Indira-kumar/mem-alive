# Engineering loop

The repository has a durable feature-delivery loop for turning roadmap items into reviewable pull requests. The owner defines direction and acceptance; the coding agent owns routine implementation, evaluation, repair, verification, documentation, and promotion.

## Starting a feature

A request only needs two things:

```text
Implement <roadmap item or spec path>.

Goal:
<the outcome this should create>

Acceptance criteria:
- <observable behavior or quality bar>
- <another observable behavior or quality bar>
```

Add constraints only when they matter, such as a required API shape, model, compatibility target, deadline, or an instruction not to push. Otherwise the repository defaults apply automatically.

For a specification that already states the goal and acceptance criteria, the short form is enough:

```text
Implement spec/003_example.md. Loop until its acceptance criteria pass, then open or update the PR.
```

## What happens automatically

The `loop-engineer` repository skill directs the coding agent to:

1. Read the request, repository instructions, and applicable specification.
2. Inspect the existing architecture and turn acceptance criteria into verification evidence.
3. Work on a scoped feature branch while preserving unrelated changes.
4. Implement the smallest maintainable change consistent with existing code.
5. Add deterministic tests and at least one feature eval.
6. Run the eval, diagnose failures, fix root causes, and repeat until it passes.
7. Run the full test, lint, type-check, and applicable build gates.
8. Document decisions and observed live-model baselines.
9. Push focused commits and create or update a pull request for owner review.

The agent asks for input only when a choice would materially change the product, required authority is missing, or an external system prevents further progress. Normal implementation details are resolved from the codebase and reported in the handoff.

## Definition of done

A feature is done only when:

- every acceptance criterion has concrete evidence;
- deterministic tests and the relevant feature eval pass;
- lint and type checks pass;
- the package builds when its distribution or public surface changed;
- operating behavior and non-obvious decisions are documented;
- the review branch and pull request contain only intended work.

Creating code is not sufficient. A live-model feature also needs a recorded live run against the requested model when that model is available.

## Staying on top of the technical work

Each final handoff leads with the outcome and includes:

- the public behavior that changed;
- the important design decisions and their rationale;
- exact test and eval results, including live model and metrics;
- links to the principal files and pull request;
- known limitations or follow-up work that was deliberately left out.

Review feedback compounds over time. General repository rules belong in `AGENTS.md`, repeatable engineering steps belong in the repository skill, and objective invariants belong in automated checks. This keeps the loop improving without turning every past preference into permanent process.
