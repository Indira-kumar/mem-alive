"""Live PageIndex section and page navigation without the chat agent."""

import asyncio
import tempfile
from pathlib import Path
from time import monotonic

from mem_alive.documents import PageIndexDocuments


async def evaluate() -> None:
    backend = {"api_key": "ollama", "base_url": "http://127.0.0.1:11434/v1"}
    fixture = Path(__file__).parent / "fixtures/project_northstar.pdf"
    with tempfile.TemporaryDirectory() as directory:
        documents = PageIndexDocuments.local(
            directory,
            index_model="openai/llama3.2",
            chat_model="openai/llama3.2",
            index_backend=backend,
            chat_backend=backend,
        )
        started = monotonic()
        doc_id = await asyncio.wait_for(documents.index_pdf(fixture), timeout=180)
        tree = await asyncio.wait_for(documents.structure(doc_id), timeout=30)
        pages = await asyncio.wait_for(documents.pages(doc_id, "2,4"), timeout=30)
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
    asyncio.run(evaluate())
