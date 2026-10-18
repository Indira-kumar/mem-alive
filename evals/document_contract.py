"""Adapter contract eval; live PageIndex answer quality is a separate release gate."""

import asyncio
import tempfile
from pathlib import Path

from mem_alive.documents import PageIndexDocuments


class StubClient:
    def __init__(self):
        self.out_of_scope = False
        self.uncited = False
        self.deleted = None

    def submit_document(self, path: str) -> dict:
        return {"doc_id": "document-1"}

    def chat(self, question: str, *, doc_id: str, citations: bool, extra_body: dict) -> str:
        return 'Supported answer. <cite doc="report.pdf" page="2"/>'

    def get_citations(self, answer: str) -> list[dict]:
        if self.uncited:
            return []
        return [
            {
                "doc_id": "other" if self.out_of_scope else "document-1",
                "document": "report.pdf",
                "page": 2,
            }
        ]

    def list_documents(self) -> dict:
        return {"documents": [{"id": "document-1"}], "total": 1}

    def delete_document(self, doc_id: str) -> None:
        self.deleted = doc_id


async def evaluate() -> None:
    stub = StubClient()
    documents = PageIndexDocuments(stub)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.pdf"
        path.write_bytes(b"fixture")
        doc_id = await documents.index_pdf(path)
        result = await documents.ask(doc_id, "What does the report say?")
        checks = {
            "document identity": doc_id == "document-1",
            "page citation": len(result.citations) == 1 and result.citations[0].page == 2,
        }
        stub.out_of_scope = True
        try:
            await documents.ask(doc_id, "What does the report say?")
            checks["scope violation rejected"] = False
        except ValueError:
            checks["scope violation rejected"] = True
        stub.out_of_scope = False
        stub.uncited = True
        try:
            await documents.ask(doc_id, "What does the report say?")
            checks["uncited answer rejected"] = False
        except ValueError:
            checks["uncited answer rejected"] = True
        checks["document listing"] = (await documents.list_documents())["total"] == 1
        await documents.delete_document(doc_id)
        checks["document deletion"] = stub.deleted == doc_id
    passed = sum(checks.values())
    print(f"document_contract: {passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
