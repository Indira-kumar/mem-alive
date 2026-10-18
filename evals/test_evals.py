"""Mutation checks for the graders; a broken product must not get a green report."""

from copy import deepcopy

import pytest

from evals.coding import load_cases as load_coding_cases
from evals.coding import score as score_coding
from evals.contracts import load_cases as load_core_contracts
from evals.contracts import score as score_contracts
from evals.core import probes, run, score_retrieval
from evals.retrieval import load_cases as load_retrieval_cases
from evals.retrieval import score as score_retrieval_cases
from evals.retrieval import summarize as summarize_retrieval


def _gold_contract_run(case: dict) -> dict:
    def make_output(expect: dict) -> dict:
        evidence = {}
        records = {}
        for key, gold in expect.get("records", {}).items():
            links = []
            for source in gold["sources"]:
                link = f"{key}:{source}"
                evidence[link] = {"source_event_id": source}
                links.append(link)
            records[key] = {
                "type": gold["type"],
                "state": gold["state"],
                "evidence_ids": links,
            }
        output = {"evidence": evidence, "records": records}
        for field in ("recall", "review", "inspectable"):
            if field in expect:
                output[field] = deepcopy(expect[field])
        return output

    return {
        "case_id": case["id"],
        "final": make_output(case["expect"]),
        "after": {
            event_id: make_output(expected)
            for event_id, expected in case.get("expect_after", {}).items()
        },
    }


def _gold_coding_runs(case: dict) -> list[dict]:
    condition = {
        "model": "test-model",
        "tools_hash": "tools-1",
        "repo_state_hash": "checkout-1",
        "task": case["session_b_task"],
    }
    return [
        {
            "case_id": case["id"],
            "mode": "off",
            "task_passed": False,
            "condition": deepcopy(condition),
        },
        {
            "case_id": case["id"],
            "mode": "on",
            "task_passed": True,
            "condition": deepcopy(condition),
            "delivered_context": deepcopy(case["required_context"]),
            "user_restatements": 0,
        },
    ]


def test_core_probe_corpus_and_grader_rejects_wrong_context():
    cases = probes()
    assert {case.category for case in cases} >= {
        "scope",
        "versioning",
        "reconciliation",
        "episode",
        "abstention",
        "ranking",
    }
    for case in cases:
        good_context = tuple(case.required)
        if case.min_results and not good_context:
            good_context = (next(iter(case.allowed)),)
        good = score_retrieval(case, good_context)
        assert good.passed
        bad = score_retrieval(case, good_context + ("unrelated memory",))
        assert not bad.passed
        assert bad.context_precision < good.context_precision or not case.required


async def test_current_core_probe_run_reports_gaps_without_relabeling_them():
    results = await run()
    assert len(results) == len(probes())
    assert all(
        result.passed
        for result in results
        if result.name
        in {
            "namespace-isolation",
            "metadata-isolation",
            "semantic-version-visibility",
            "episodes-remain-distinct",
            "unrelated-query-abstains",
        }
    )
    for result in results:
        if not result.passed:
            assert result.missing or result.unexpected or result.limit_exceeded


def test_core_contracts_cover_product_invariants_and_pass_gold_outputs():
    cases = load_core_contracts()
    assert len(cases) == 11
    assert all(case["events"] and case["queries"] for case in cases)
    for case in cases:
        event_ids = {event["id"] for event in case["events"]}
        query_ids = {query["id"] for query in case["queries"]}
        assert set(case["expect"]["recall"]) <= query_ids
        assert set(case.get("expect_after", {})) <= event_ids
        for record in case["expect"]["records"].values():
            assert set(record["sources"]) <= event_ids
    assert all(
        item.passed for item in score_contracts(cases, [_gold_contract_run(case) for case in cases])
    )


def test_core_contract_grader_catches_broken_evidence_chain_and_replay():
    cases = load_core_contracts()
    runs = [_gold_contract_run(case) for case in cases]
    provenance = next(run for run in runs if run["case_id"] == "provenance-chain")
    provenance["final"]["evidence"]["billing-db:e1"]["source_event_id"] = "agent-claim"
    replay = next(run for run in runs if run["case_id"] == "event-replay-is-idempotent")
    replay["final"]["records"]["duplicate"] = deepcopy(replay["final"]["records"]["badge-color"])
    scores = {item.case_id: item for item in score_contracts(cases, runs)}
    assert not scores["provenance-chain"].passed
    assert any("provenance" in failure for failure in scores["provenance-chain"].failures)
    assert not scores["event-replay-is-idempotent"].passed


def test_core_contract_grader_catches_early_procedure_promotion():
    cases = load_core_contracts()
    runs = [_gold_contract_run(case) for case in cases]
    procedure = next(
        run for run in runs if run["case_id"] == "procedure-needs-independent-verification"
    )
    procedure["after"]["e2"]["records"]["restart-conditional"]["state"] = "established"
    score = next(
        item for item in score_contracts(cases, runs) if item.case_id == procedure["case_id"]
    )
    assert not score.passed
    assert any("e2" in failure for failure in score.failures)


def test_coding_grader_requires_task_outcome_and_correct_context():
    cases = load_coding_cases()
    assert len(cases) == 8
    assert all(case["session_a"] and case["session_b_task"] and case["oracle"] for case in cases)
    runs = [run for case in cases for run in _gold_coding_runs(case)]
    assert all(item.passed for item in score_coding(cases, runs))

    broken = deepcopy(runs)
    target = next(run for run in broken if run["case_id"] == cases[0]["id"] and run["mode"] == "on")
    target["task_passed"] = False
    target["delivered_context"] = deepcopy(cases[0]["required_context"])
    scores = {item.case_id: item for item in score_coding(cases, broken)}
    assert not scores[cases[0]["id"]].passed
    assert not scores[cases[0]["id"]].task_passed


def test_coding_grader_catches_stale_context_and_restatements():
    cases = load_coding_cases()
    runs = [run for case in cases for run in _gold_coding_runs(case)]
    changed = next(
        run for run in runs if run["case_id"] == "changed-decision" and run["mode"] == "on"
    )
    changed["delivered_context"].append({"id": "redis-cache-stale", "status": "active"})
    changed["user_restatements"] = 1
    result = next(item for item in score_coding(cases, runs) if item.case_id == "changed-decision")
    assert not result.passed
    assert result.forbidden_context_count == 1
    assert result.unexpected_context_count == 1
    assert result.user_restatements == 1


def test_coding_grader_rejects_missing_or_duplicate_pair():
    cases = load_coding_cases()
    runs = [run for case in cases for run in _gold_coding_runs(case)]
    with pytest.raises(ValueError, match="missing paired run"):
        score_coding(cases, runs[:-1])
    with pytest.raises(ValueError, match="duplicate run"):
        score_coding(cases, runs + [runs[0]])


def test_coding_grader_checks_pair_parity_and_repeated_trials():
    cases = load_coding_cases()
    runs = [run for case in cases for run in _gold_coding_runs(case)]
    repeat = deepcopy(runs)
    for item in repeat:
        item["trial_id"] = "2"
    scores = score_coding(cases, runs + repeat)
    assert len(scores) == 2 * len(cases)
    mismatch = deepcopy(runs)
    mismatch[1]["condition"]["model"] = "different-model"
    with pytest.raises(ValueError, match="paired conditions differ"):
        score_coding(cases, mismatch)


def test_held_out_retrieval_grader_catches_near_match_and_scope_leak():
    cases = load_retrieval_cases()
    assert len(cases) == 7
    assert any(not case["required"] for case in cases)
    assert {case["type"] for case in cases} == {"semantic", "episodic", "procedural"}
    for case in cases:
        memory_ids = {memory["id"] for memory in case["memories"]}
        required, optional, forbidden = (
            set(case[name]) for name in ("required", "optional", "forbidden")
        )
        assert required | optional | forbidden <= memory_ids
        assert not (required & optional or required & forbidden or optional & forbidden)
    runs = [
        {
            "case_id": case["id"],
            "hit_ids": case["required"],
            "context_tokens": 24,
            "latency_ms": 2.5,
        }
        for case in cases
    ]
    good_scores = score_retrieval_cases(cases, runs)
    assert all(item.passed for item in good_scores)
    assert summarize_retrieval(good_scores)["type:episodic"]["required_recall"] == 1.0
    broken = deepcopy(runs)
    broken[0]["hit_ids"] = ["target", "other-project"]
    negative = next(run for run in broken if run["case_id"] == "irrelevant-near-match-abstains")
    negative["hit_ids"] = ["near"]
    scores = {item.case_id: item for item in score_retrieval_cases(cases, broken)}
    assert not scores["exact-entity-and-path"].passed
    assert not scores["irrelevant-near-match-abstains"].passed
    assert scores["irrelevant-near-match-abstains"].wrong_memory_count == 1
