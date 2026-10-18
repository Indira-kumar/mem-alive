"""Score paired fresh-session coding runs from a host-specific adapter.

The adapter must execute each task oracle against the produced workspace;
agent self-report is not a task outcome.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

CASES_PATH = Path(__file__).with_name("coding_cases.json")


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    trial_id: str
    passed: bool
    task_passed: bool
    baseline_task_passed: bool
    required_context_recall: float
    forbidden_context_count: int
    unexpected_context_count: int
    context_precision: float
    user_restatements: int
    regression: bool


def load_cases() -> list[dict]:
    """Read the fresh-session corpus with a stable encoding on every platform."""
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def score(cases: list[dict], runs: list[dict]) -> list[CaseScore]:
    """Reject unpaired trials and compare task, tools, model, and checkout state."""
    expected_ids = {case["id"] for case in cases}
    if len(expected_ids) != len(cases):
        raise ValueError("duplicate case ID")
    indexed: dict[tuple[str, str, str], dict] = {}
    trials: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        trial = run.get("trial_id", "1")
        key = (run["case_id"], trial, run["mode"])
        if run["case_id"] not in expected_ids or run["mode"] not in {"on", "off"}:
            raise ValueError(f"unknown case or mode: {key}")
        if not isinstance(trial, str) or not trial:
            raise ValueError(f"invalid trial ID: {key}")
        if key in indexed:
            raise ValueError(f"duplicate run: {key}")
        if type(run.get("task_passed")) is not bool:
            raise ValueError(f"task_passed must be a boolean: {key}")
        indexed[key] = run
        trials[run["case_id"]].add(trial)

    scores = []
    for case in cases:
        case_id = case["id"]
        if not trials[case_id]:
            raise ValueError(f"missing paired run for {case_id}")
        for trial in sorted(trials[case_id]):
            try:
                off = indexed[(case_id, trial, "off")]
                on = indexed[(case_id, trial, "on")]
            except KeyError as exc:
                raise ValueError(f"missing paired run for {case_id}, trial {trial}") from exc
            scores.append(_score_pair(case, trial, off, on))
    return scores


def _score_pair(case: dict, trial: str, off: dict, on: dict) -> CaseScore:
    """Reject incomplete measurements before judging context and task outcome."""
    case_id = case["id"]
    condition = on.get("condition")
    fields = {"model", "tools_hash", "repo_state_hash", "task"}
    if (
        not isinstance(condition, dict)
        or condition != off.get("condition")
        or not fields <= condition.keys()
        or any(not condition[field] for field in fields)
        or condition["task"] != case["session_b_task"]
    ):
        raise ValueError(f"paired conditions differ or omit task setup: {case_id}, trial {trial}")
    if off.get("delivered_context"):
        raise ValueError(f"memory-off run received context: {case_id}")
    if "delivered_context" not in on or "user_restatements" not in on:
        raise ValueError(f"missing memory-on measurements: {case_id}")
    delivered = on["delivered_context"]
    if len({(item["id"], item["status"]) for item in delivered}) != len(delivered):
        raise ValueError(f"duplicate delivered context: {case_id}")
    supplied = {(item["id"], item["status"]) for item in delivered}
    required = {(item["id"], item["status"]) for item in case["required_context"]}
    forbidden = {(item["id"], item["status"]) for item in case["forbidden_context"]}
    permitted = {(item["id"], item["status"]) for item in case.get("permitted_context", [])}
    missing = required - supplied
    wrong = len(forbidden & supplied)
    unexpected = len(supplied - required - permitted)
    restatements = on["user_restatements"]
    if not isinstance(restatements, int) or isinstance(restatements, bool) or restatements < 0:
        raise ValueError(f"invalid user_restatements: {case_id}")
    regression = off["task_passed"] and not on["task_passed"]
    if required:
        recall = 1 - len(missing) / len(required)
    elif permitted:
        recall = 1.0
    else:
        recall = float(not supplied)
    return CaseScore(
        case_id=case_id,
        trial_id=trial,
        passed=on["task_passed"]
        and not missing
        and not wrong
        and not unexpected
        and not restatements,
        task_passed=on["task_passed"],
        baseline_task_passed=off["task_passed"],
        required_context_recall=recall,
        forbidden_context_count=wrong,
        unexpected_context_count=unexpected,
        context_precision=len((required | permitted) & supplied) / len(supplied)
        if supplied
        else float(not required),
        user_restatements=restatements,
        regression=regression,
    )


def summarize(scores: list[CaseScore]) -> dict:
    """Expose per-case repeat variance instead of one aggregate pass rate."""
    groups: dict[str, list[CaseScore]] = defaultdict(list)
    for item in scores:
        groups[item.case_id].append(item)
    return {
        case_id: {
            "trials": len(items),
            "pass_rate": mean(item.passed for item in items),
            "memory_on_task_success": mean(item.task_passed for item in items),
            "memory_off_task_success": mean(item.baseline_task_passed for item in items),
            "regressions": sum(item.regression for item in items),
        }
        for case_id, items in groups.items()
    }


def main() -> None:
    """Print every trial and fail the command when any paired run fails."""
    parser = argparse.ArgumentParser(description="Score paired coding-agent eval runs")
    parser.add_argument("results", type=Path, help="JSON file containing paired run records")
    args = parser.parse_args()
    scores = score(load_cases(), json.loads(args.results.read_text(encoding="utf-8"))["runs"])
    print(
        json.dumps(
            {"scores": [asdict(item) for item in scores], "summary": summarize(scores)}, indent=2
        )
    )
    raise SystemExit(0 if all(item.passed for item in scores) else 1)


if __name__ == "__main__":
    main()
