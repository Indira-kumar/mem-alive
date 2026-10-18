import json

import pytest
from mem_alive.documents import PageIndexDocuments


class FakePageIndexClient:
    def __init__(
        self, citation_doc_id: str = "doc-1", cited: bool = True, pages=None, answer=None
    ):
        self.citation_doc_id = citation_doc_id
        self.cited = cited
        self.pages = pages or [{"page_index": 1, "markdown": "Unrelated page."}]
        self.answer = answer or 'Answer <cite doc="source.pdf" page="3"/>'
        self.deleted = None
        self.last_question = None

    def submit_document(self, path: str) -> dict:
        assert path.endswith(".pdf")
        return {"doc_id": "doc-1"}

    def chat(self, question: str, *, doc_id: str, citations: bool, extra_body: dict) -> str:
        self.last_question = (question, doc_id, citations, extra_body)
        return self.answer

    def get_citations(self, answer: str) -> list[dict]:
        if not self.cited:
            return []
        return [{"doc_id": self.citation_doc_id, "document": "source.pdf", "page": 3}]

    def get_document(self, doc_id: str) -> dict:
        return {"name": "source.pdf", "pageNum": len(self.pages)}

    def get_page_content(self, doc_id: str, pages: str) -> list[dict]:
        return self.pages

    def list_documents(self) -> dict:
        return {"documents": [{"id": "doc-1"}], "total": 1}

    def delete_document(self, doc_id: str) -> None:
        self.deleted = doc_id


async def test_index_and_ask_with_page_citation(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"placeholder")
    client = FakePageIndexClient()
    documents = PageIndexDocuments(client)

    doc_id = await documents.index_pdf(path)
    response = await documents.ask(doc_id, "What is the answer?")

    assert client.last_question == ("What is the answer?", "doc-1", True, {"temperature": 0})
    assert response.citations[0].doc_id == doc_id
    assert response.citations[0].page == 3


async def test_wrong_document_citation_is_rejected():
    documents = PageIndexDocuments(FakePageIndexClient(citation_doc_id="other"))
    with pytest.raises(ValueError, match="outside"):
        await documents.ask("doc-1", "Question")


async def test_uncited_answer_is_rejected():
    documents = PageIndexDocuments(FakePageIndexClient(cited=False))
    with pytest.raises(ValueError, match="no verifiable"):
        await documents.ask("doc-1", "Question")


async def test_uncited_numeric_answer_is_verified_against_one_page():
    client = FakePageIndexClient(
        cited=False,
        pages=[
            {"page_index": 1, "markdown": "Old backup interval was 45 minutes."},
            {
                "page_index": 2,
                "markdown": "The backup interval for Project Northstar is 17 minutes.",
            },
        ],
        answer="The backup interval for Project Northstar is 17 minutes.",
    )
    result = await PageIndexDocuments(client).ask("doc-1", "Current backup interval?")
    assert result.citations[0].page == 2
    assert result.citations[0].verification == "text_span"


async def test_uncited_numeric_answer_rejects_negated_source():
    client = FakePageIndexClient(
        cited=False,
        pages=[
            {
                "page_index": 1,
                "markdown": "The backup interval for Project Northstar is not 17 minutes.",
            }
        ],
        answer="The backup interval for Project Northstar is 17 minutes.",
    )
    with pytest.raises(ValueError, match="no verifiable"):
        await PageIndexDocuments(client).ask("doc-1", "Current backup interval?")


async def test_non_pdf_input_is_rejected(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("content")
    with pytest.raises(ValueError, match="PDF"):
        await PageIndexDocuments(FakePageIndexClient()).index_pdf(path)


async def test_document_lifecycle_routes_to_scoped_client():
    client = FakePageIndexClient()
    documents = PageIndexDocuments(client)
    assert (await documents.list_documents())["total"] == 1
    await documents.delete_document("doc-1")
    assert client.deleted == "doc-1"


async def test_retries_pageindex_tool_error_once():
    class FlakyClient(FakePageIndexClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def chat(self, question: str, *, doc_id: str, citations: bool, extra_body: dict) -> str:
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {"name": "get_document", "parameters": {"errorCode": "INVALID_INPUT"}}
                )
            return super().chat(question, doc_id=doc_id, citations=citations, extra_body=extra_body)

    client = FlakyClient()
    await PageIndexDocuments(client).ask("doc-1", "Question")
    assert client.calls == 2
