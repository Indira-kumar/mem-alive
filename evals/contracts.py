"""Score normalized outputs from a future core adapter against behavior contracts.

The adapter translates its own IDs into fixture keys. Evidence links are scored
by traversing the adapter's evidence records to source event IDs.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CASES_PATH = Path(__file__).with_name("core_contracts.json")


@dataclass(frozen=True)
class ContractScore:
    case_id: str
    passed: bool
    failures: tuple[str, ...]


def load_cases() -> list[dict]:
    """Read the implementation-neutral core contract corpus."""
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _compare(expected: dict, observed: dict, prefix: str) -> list[str]:
    """Compare visible state and trace each memory through evidence to sources."""
    failures = []
    for field in ("recall", "review", "inspectable", "memory_count", "evidence_count"):
        if field not in expected:
            continue
        if field == "recall":
            for query_id, gold in expected[field].items():
                actual = observed.get("recall", {}).get(query_id)
                if actual is None or len(actual) != len(set(actual)) or set(actual) != set(gold):
                    failures.append(f"{prefix}.recall.{query_id}")
        elif field in {"review", "inspectable"}:
            actual = observed.get(field)
            if (
                actual is None
                or len(actual) != len(set(actual))
                or set(actual) != set(expected[field])
            ):
                failures.append(f"{prefix}.{field}")
        elif (
            len(observed.get("records" if field == "memory_count" else "evidence", {}))
            != expected[field]
        ):
            failures.append(f"{prefix}.{field}")

    if "records" not in expected:
        return failures
    gold_records = expected["records"]
    actual_records = observed.get("records")
    if actual_records is None or set(actual_records) != set(gold_records):
        failures.append(f"{prefix}.records")
        return failures
    evidence = observed.get("evidence", {})
    for key, gold in gold_records.items():
        record = actual_records[key]
        for field in ("type", "state"):
            if record.get(field) != gold[field]:
                failures.append(f"{prefix}.records.{key}.{field}")
        links = record.get("evidence_ids", [])
        sources = [evidence.get(link, {}).get("source_event_id") for link in links]
        if (
            len(links) != len(set(links))
            or None in sources
            or len(sources) != len(set(sources))
            or set(sources) != set(gold["sources"])
        ):
            failures.append(f"{prefix}.records.{key}.provenance")
    return failures


def score(cases: list[dict], runs: list[dict]) -> list[ContractScore]:
    """Reject missing cases and report each final or intermediate mismatch."""
    expected_ids = {case["id"] for case in cases}
    if len(expected_ids) != len(cases):
        raise ValueError("duplicate contract case ID")
    indexed = {}
    for run in runs:
        case_id = run["case_id"]
        if case_id not in expected_ids or case_id in indexed:
            raise ValueError(f"unknown or duplicate contract run: {case_id}")
        indexed[case_id] = run
    scores = []
    for case in cases:
        case_id = case["id"]
        if case_id not in indexed:
            raise ValueError(f"missing contract run: {case_id}")
        run = indexed[case_id]
        failures = _compare(case["expect"], run.get("final", {}), f"{case_id}.final")
        for event_id, expected in case.get("expect_after", {}).items():
            failures.extend(
                _compare(expected, run.get("after", {}).get(event_id, {}), f"{case_id}.{event_id}")
            )
        scores.append(ContractScore(case_id, not failures, tuple(failures)))
    return scores


def main() -> None:
    """Score adapter JSON and return a failing status for any broken contract."""
    parser = argparse.ArgumentParser(description="Score normalized core behavior contract runs")
    parser.add_argument("results", type=Path, help="JSON file containing adapter run records")
    args = parser.parse_args()
    scores = score(load_cases(), json.loads(args.results.read_text(encoding="utf-8"))["runs"])
    print(json.dumps({"scores": [asdict(item) for item in scores]}, indent=2))
    raise SystemExit(0 if all(item.passed for item in scores) else 1)


if __name__ == "__main__":
    main()
