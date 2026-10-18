"""Score held-out retrieval results without assuming a search implementation."""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

CASES_PATH = Path(__file__).with_name("retrieval_cases.json")


@dataclass(frozen=True)
class RetrievalScore:
    case_id: str
    memory_type: str
    scope: str
    passed: bool
    required_recall: float
    context_precision: float
    wrong_memory_count: int
    abstained: bool
    context_tokens: int
    latency_ms: float


def load_cases() -> list[dict]:
    """Read held-out queries independently of the system under test."""
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def score(cases: list[dict], runs: list[dict]) -> list[RetrievalScore]:
    """Count required, optional, and wrong hits without a numeric cutoff."""
    expected_ids = {case["id"] for case in cases}
    if len(expected_ids) != len(cases):
        raise ValueError("duplicate retrieval case ID")
    indexed = {}
    for run in runs:
        case_id = run["case_id"]
        if case_id not in expected_ids or case_id in indexed:
            raise ValueError(f"unknown or duplicate retrieval run: {case_id}")
        indexed[case_id] = run

    scores = []
    for case in cases:
        case_id = case["id"]
        if case_id not in indexed:
            raise ValueError(f"missing retrieval run: {case_id}")
        run = indexed[case_id]
        hits = run["hit_ids"]
        if len(hits) != len(set(hits)):
            raise ValueError(f"duplicate hit: {case_id}")
        if (
            type(run.get("context_tokens")) is not int
            or run["context_tokens"] < 0
            or type(run.get("latency_ms")) not in (int, float)
            or not math.isfinite(run["latency_ms"])
            or run["latency_ms"] < 0
        ):
            raise ValueError(f"invalid size or latency: {case_id}")
        required = set(case["required"])
        allowed = required | set(case["optional"])
        found = set(hits)
        wrong = len(found - allowed)
        if required:
            recall = len(found & required) / len(required)
        elif case["optional"]:
            recall = 1.0
        else:
            recall = float(not found)
        precision = len(found & allowed) / len(hits) if hits else float(not required)
        scores.append(
            RetrievalScore(
                case_id=case_id,
                memory_type=case["type"],
                scope=case["scope"],
                passed=required <= found and wrong == 0,
                required_recall=recall,
                context_precision=precision,
                wrong_memory_count=wrong,
                abstained=not hits,
                context_tokens=run["context_tokens"],
                latency_ms=run["latency_ms"],
            )
        )
    return scores


def summarize(scores: list[RetrievalScore]) -> dict:
    """Keep type and scope slices visible; report latency without a pass cutoff."""
    groups: dict[str, list[RetrievalScore]] = defaultdict(list)
    for item in scores:
        groups[f"type:{item.memory_type}"].append(item)
        groups[f"scope:{item.scope}"].append(item)

    summary = {}
    for key, items in groups.items():
        latencies = sorted(item.latency_ms for item in items)

        def percentile(p: float) -> float:
            """Use the nearest observed rank for small eval corpora."""
            return latencies[math.ceil(p * len(latencies)) - 1]

        summary[key] = {
            "cases": len(items),
            "pass_rate": mean(item.passed for item in items),
            "required_recall": mean(item.required_recall for item in items),
            "context_precision": mean(item.context_precision for item in items),
            "wrong_memories": sum(item.wrong_memory_count for item in items),
            "mean_context_tokens": mean(item.context_tokens for item in items),
            "latency_ms_p50": percentile(0.50),
            "latency_ms_p95": percentile(0.95),
            "latency_ms_p99": percentile(0.99),
        }
    return summary


def main() -> None:
    """Print per-case and per-scope metrics for adapter results."""
    parser = argparse.ArgumentParser(description="Score held-out retrieval runs")
    parser.add_argument("results", type=Path, help="JSON file containing adapter run records")
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
