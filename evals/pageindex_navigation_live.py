"""Live PageIndex CLI navigation across separate agent processes."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import monotonic


def evaluate() -> None:
    fixture = Path(__file__).parent / "fixtures/project_northstar.pdf"
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "memory.db"

        def command(*arguments: str) -> object:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mem_alive.cli",
                    "--db",
                    str(database),
                    "--namespace",
                    "project",
                    *arguments,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            return json.loads(completed.stdout)

        started = monotonic()
        indexed = command("index-pdf", str(fixture))
        assert isinstance(indexed, dict)
        doc_id = indexed["doc_id"]
        tree = command("doc-tree", doc_id)
        pages = command("doc-pages", doc_id, "2,4")
        assert isinstance(tree, list) and isinstance(pages, list)
        by_page = {page["page_index"]: page["markdown"] for page in pages}
        checks = {
            "nonempty section tree": bool(tree),
            "current page text": "17 minutes" in by_page.get(2, ""),
            "retired page text": "45 minutes" in by_page.get(4, ""),
        }
        print(f"pageindex_navigation: {sum(checks.values())}/{len(checks)} "
              f"checks passed in {monotonic() - started:.1f}s")
        if not all(checks.values()):
            print(f"checks={checks} tree={tree} pages={pages}")
            raise SystemExit(1)


if __name__ == "__main__":
    evaluate()
