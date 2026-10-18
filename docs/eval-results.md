# v1 candidate evaluation record

Date: 2026-10-18. Environment: local macOS, Python 3.12, Ollama with `embeddinggemma` (768-dimensional embeddings) and `llama3.2` (3.2B, Q4_K_M). PageIndex SDK 0.4.1 runs locally against Ollama's OpenAI-compatible endpoint. No cloud model or paid API was used.

## Deterministic gates

| Gate | Result |
| --- | --- |
| Full pytest suite | 65 passed |
| Ruff | Passed |
| ty | Passed |
| Source distribution and wheel | Built successfully |
| Safe semantic policy eval | 3/3 |
| SQLite restart, history, backup, and browse eval | 7/7 |
| Federated retrieval and candidate isolation eval | 5/5 |
| PageIndex adapter contract eval | 8/8 |
| CLI separate-process eval | 7/7 |

The command lines are in [operations](operations.md). These tests use fake embeddings or a temporary SQLite file and have no live-model quality implications.

## Live memory baselines

`embeddinggemma` with the measured 0.7 recall cutoff ranked the expected record first for 5/5 answerable paraphrases. All 3 unanswerable queries returned no memory. The weakest correct raw cosine score was 0.734; the strongest negative was 0.579. This eight-query synthetic set is small and cannot establish performance on personal work.

`llama3.2` at temperature 0 with Ollama's JSON-schema response format passed 6/6 conversation extraction cases: preference, completed event, repeatable procedure, user correction, assistant-only suggestion, and small talk. Invalid assistant source indices were discarded before storage. The set does not measure long conversations, multilingual input, prompt injection, or secret omission reliability.

The JSON CLI passed 7/7 live cross-process checks: recall, correction, history, namespace isolation, backup restore, conversation ingestion, and history purge via `forget`.

Direct PageIndex navigation passed 3/3 live checks on the synthetic PDF: a nonempty section tree and the expected source text on pages 2 and 4. Indexing plus navigation took 5.3 seconds. This gives an agent a route to inspect source pages even when PageIndex chat fails.

## Live PageIndex baseline

The fixture is a four-page text PDF in `evals/fixtures/project_northstar.pdf`. It has a current 17-minute backup interval on page 2, a retired 45-minute interval on page 4, and no monthly revenue. The release eval asks both answerable questions and the unsupported question, and requires correct page citations or a unique numeric claim verified against the page text.

One run passed: 2/2 correct answers, 2/2 correct page citations, and the unsupported query was withheld. Indexing took 4.7 seconds; query times were 10.5, 28.1, and 38.9 seconds. An earlier run under the same temperature setting failed both answerable questions after 52.4 and 50.5 seconds; another response exposed a PageIndex tool-argument error. A single bounded retry was added for that specific error, and one subsequent full run passed. This is **not a stable quality baseline**. The adapter withholds answers without a citation or a unique verified numeric source span, so its failure mode is visible rather than silently returning an ungrounded answer.

A repeat run after opening the draft PR failed: the first supported answer and citation were correct, the second returned an invalid `get_document` tool call, and the unsupported query exceeded the SDK's 10-turn limit. The adapter now surfaces those cases as controlled retrieval errors. The PageIndex chat quality gate remains failed until the model and retrieval path are reliable across repeated runs.

Before a v1 claim, repeat the PageIndex eval on more documents and questions, including nonnumeric answers, scanned or image-rich files where applicable, and ambiguous or outdated passages. Record answer accuracy, citation precision, abstention, p95 latency, and model/token cost. A representative private corpus and an actual agent workflow are still needed.
