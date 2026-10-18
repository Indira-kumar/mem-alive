"""Live CLI process-restart eval; requires local Ollama and installed package script."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def command(db: Path, namespace: str, *args: str, input_text: str | None = None):
    executable = Path(sys.executable).parent / "mem-alive"
    result = subprocess.run(
        [str(executable), "--db", str(db), "--namespace", namespace, *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr}")
    return json.loads(result.stdout)


def evaluate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "memory.db"
        saved = command(db, "personal", "remember", "semantic", "I prefer Python")
        memory_id = saved["id"]
        recalled = command(db, "personal", "recall", "I prefer Python")
        updated = command(db, "personal", "update", memory_id, "I prefer Rust")
        history = command(db, "personal", "history", memory_id)
        other_scope = command(db, "other", "get", memory_id)
        backup = Path(directory) / "backup.db"
        command(db, "personal", "backup", str(backup))
        backup_record = command(backup, "personal", "get", memory_id)
        ingested = command(
            db,
            "personal",
            "ingest",
            input_text=json.dumps(
                [{"role": "user", "content": "I prefer dark mode in coding tools."}]
            ),
        )
        command(db, "personal", "forget", memory_id)
        forgotten = command(db, "personal", "get", memory_id)
        forgotten_history = command(db, "personal", "history", memory_id)
        checks = {
            "cross-process recall": bool(recalled) and recalled[0]["id"] == memory_id,
            "explicit correction": updated["content"] == "I prefer Rust",
            "auditable history": [event["action"] for event in history] == ["add", "update"],
            "namespace isolation": other_scope is None,
            "backup restore": backup_record["content"] == "I prefer Rust",
            "conversation ingestion": len(ingested) == 1
            and "dark mode" in ingested[0]["content"].lower(),
            "forget purges active and history": forgotten is None and forgotten_history == [],
        }
    passed = sum(checks.values())
    print(f"cli_roundtrip: {passed}/{len(checks)} checks passed")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    evaluate()
