"""Live PageIndex/Ollama eval on a four-page, text-based synthetic PDF."""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from mem_alive.documents import DocumentRetrievalError, PageIndexDocuments, UncitedAnswerError


async def evaluate() -> None:
    fixture = Path(__file__).parent / "fixtures/project_northstar.pdf"
    backend = {"api_key": "ollama", "base_url": "http://127.0.0.1:11434/v1"}
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
        indexed_seconds = monotonic() - started
        cases = [
            ("What is the current backup interval for Project Northstar?", "17 minutes", 2),
            ("What backup interval was retired in 2024?", "45 minutes", 4),
        ]
        correct = 0
        cited = 0
        query_seconds = []
        for question, expected, page in cases:
            query_started = monotonic()
            try:
                answer = await asyncio.wait_for(documents.ask(doc_id, question), timeout=120)
                correct += expected in answer.text
                cited += any(citation.page == page for citation in answer.citations)
                print(f"question={question!r} answer={answer.text!r} citations={answer.citations}")
            except UncitedAnswerError as exc:
                print(f"question={question!r} rejected={exc} raw_answer={exc.answer!r}")
            except DocumentRetrievalError as exc:
                print(f"question={question!r} failed={exc}")
            query_seconds.append(round(monotonic() - query_started, 1))
        query_started = monotonic()
        try:
            unsupported = await asyncio.wait_for(
                documents.ask(doc_id, "What was Project Northstar's monthly revenue?"),
                timeout=120,
            )
            abstained = "not" in unsupported.text.lower() and "revenue" in unsupported.text.lower()
            print(f"unsupported_answer={unsupported.text!r}")
        except UncitedAnswerError as exc:
            abstained = True
            print(f"unsupported_rejected={exc} raw_answer={exc.answer!r}")
        except DocumentRetrievalError as exc:
            abstained = False
            print(f"unsupported_failed={exc}")
        query_seconds.append(round(monotonic() - query_started, 1))
        print(
            f"model=ollama/llama3.2 date={datetime.now(timezone.utc).date()} "
            f"index_seconds={indexed_seconds:.1f} "
            f"answer_correct={correct}/{len(cases)} citation_correct={cited}/{len(cases)} "
            f"unsupported_safe={abstained} query_seconds={query_seconds}"
        )
        if correct != len(cases) or cited != len(cases) or not abstained:
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
