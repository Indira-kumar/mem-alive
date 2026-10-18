"""Optional PageIndex boundary for cited, long-document answers."""

import asyncio
import json
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentCitation:
    doc_id: str
    document: str
    page: int
    verification: str = "model"


@dataclass(frozen=True)
class DocumentAnswer:
    text: str
    citations: tuple[DocumentCitation, ...]


class UncitedAnswerError(ValueError):
    """A model answer was withheld because no source page could be verified."""

    def __init__(self, answer: str):
        super().__init__("PageIndex answer has no verifiable page citation")
        self.answer = answer


class DocumentRetrievalError(RuntimeError):
    """PageIndex could not complete a document query."""


class PageIndexDocuments:
    """Keep source-document retrieval separate from atomic memory records."""

    def __init__(self, client: Any, temperature: float = 0, max_attempts: int = 2):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._client = client
        self._temperature = temperature
        self._max_attempts = max_attempts

    @classmethod
    def local(
        cls,
        storage_path: str | Path,
        index_model: str,
        chat_model: str,
        index_backend: dict | None = None,
        chat_backend: dict | None = None,
    ) -> "PageIndexDocuments":
        """Create a local PageIndex client; requires `mem-alive[pageindex]`."""
        try:
            client_type = import_module("pageindex").PageIndexClient
        except ImportError as exc:
            raise RuntimeError("install mem-alive[pageindex] for document retrieval") from exc
        index: dict[str, Any] = {"model": index_model, "storage_path": str(storage_path)}
        chat: dict[str, Any] = {"model": chat_model}
        if index_backend is not None:
            index["backend"] = index_backend
        if chat_backend is not None:
            chat["backend"] = chat_backend
        client = client_type(index=index, chat=chat)
        return cls(client)

    async def index_pdf(self, path: str | Path) -> str:
        """Index a text-based PDF and return its PageIndex document ID."""
        document = Path(path)
        if document.suffix.lower() != ".pdf" or not document.is_file():
            raise ValueError("index_pdf requires an existing PDF file")
        result = await asyncio.to_thread(self._client.submit_document, str(document))
        return str(result["doc_id"])

    async def list_documents(self) -> dict:
        """List documents in this client's scoped PageIndex library."""
        return await asyncio.to_thread(self._client.list_documents)

    async def structure(self, doc_id: str) -> list[dict]:
        """Return PageIndex's section tree without duplicating its full text."""
        if not doc_id:
            raise ValueError("doc_id is required")
        return await asyncio.to_thread(self._client.get_document_structure, doc_id)

    async def pages(self, doc_id: str, page_spec: str) -> list[dict]:
        """Read cited source pages directly for inspection or manual navigation."""
        if not doc_id or not re.fullmatch(r"[\d,\-\s]+", page_spec):
            raise ValueError("doc_id and page numbers are required")
        return await asyncio.to_thread(self._client.get_page_content, doc_id, page_spec)

    async def delete_document(self, doc_id: str) -> None:
        """Delete one document from this client's library."""
        await asyncio.to_thread(self._client.delete_document, doc_id)

    async def ask(self, doc_id: str, question: str) -> DocumentAnswer:
        """Answer from one document and expose its resolved page citations."""
        if not doc_id or not question.strip():
            raise ValueError("doc_id and question are required")
        for _ in range(self._max_attempts):
            try:
                answer = await asyncio.to_thread(
                    self._client.chat,
                    question,
                    doc_id=doc_id,
                    citations=True,
                    extra_body={"temperature": self._temperature},
                )
            except Exception as exc:
                raise DocumentRetrievalError(f"PageIndex query failed: {exc}") from exc
            try:
                payload = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                break
            details = payload.get("parameters") if isinstance(payload, dict) else None
            tool_error = (isinstance(details, dict) and bool(details.get("errorCode"))) or (
                '"errorCode"' in answer and '"parameters"' in answer
            )
            if not tool_error:
                break
        else:
            raise DocumentRetrievalError("PageIndex repeatedly returned an invalid tool call")
        raw_citations = await asyncio.to_thread(self._client.get_citations, answer)
        citations = tuple(
            DocumentCitation(
                doc_id=str(item["doc_id"]),
                document=str(item["document"]),
                page=int(item["page"]),
            )
            for item in raw_citations
        )
        if any(citation.doc_id != doc_id for citation in citations):
            raise ValueError("PageIndex answer cited a document outside the requested scope")
        if answer.strip() and not citations:
            citations = await self._verify_numeric_answer(doc_id, answer)
            if not citations:
                raise UncitedAnswerError(answer)
        return DocumentAnswer(text=answer, citations=citations)

    async def _verify_numeric_answer(
        self, doc_id: str, answer: str
    ) -> tuple[DocumentCitation, ...]:
        """Accept an uncited numeric answer only when one source page supports its words."""
        answer_text = re.sub(r"<cite\b[^>]*>", "", answer, flags=re.IGNORECASE)
        answer_tokens = re.findall(r"\w+", answer_text.casefold())
        numbers = {token for token in answer_tokens if any(char.isdigit() for char in token)}
        if not numbers:
            return ()
        stopwords = {
            "the", "a", "an", "is", "was", "were", "in", "for", "of", "to",
            "that", "this", "as", "and", "or", "at", "on", "by", "with", "it", "from",
        }
        content_tokens = {token for token in answer_tokens if token not in stopwords}
        negations = {"not", "never", "no"}
        document = await asyncio.to_thread(self._client.get_document, doc_id)
        pages = await asyncio.to_thread(
            self._client.get_page_content, doc_id, f"1-{document['pageNum']}"
        )
        matches = []
        for page in pages:
            page_tokens = re.findall(r"\w+", page["markdown"].casefold())
            if not numbers.issubset(page_tokens):
                continue
            if (negations & set(answer_tokens)) != (negations & set(page_tokens)):
                continue
            longest = 0
            for answer_start, token in enumerate(answer_tokens):
                if token not in numbers:
                    continue
                for page_start, page_token in enumerate(page_tokens):
                    if page_token != token:
                        continue
                    left = right = 0
                    while (
                        answer_start - left - 1 >= 0
                        and page_start - left - 1 >= 0
                        and answer_tokens[answer_start - left - 1]
                        == page_tokens[page_start - left - 1]
                    ):
                        left += 1
                    while (
                        answer_start + right + 1 < len(answer_tokens)
                        and page_start + right + 1 < len(page_tokens)
                        and answer_tokens[answer_start + right + 1]
                        == page_tokens[page_start + right + 1]
                    ):
                        right += 1
                    longest = max(longest, left + right + 1)
            coverage = len(content_tokens & set(page_tokens)) / len(content_tokens)
            if longest >= 6 or (len(content_tokens) >= 4 and coverage >= 0.8):
                matches.append(
                    (max(longest / len(answer_tokens), coverage), int(page["page_index"]))
                )
        matches.sort(reverse=True)
        if not matches or (len(matches) > 1 and matches[0][0] == matches[1][0]):
            return ()
        return (
            DocumentCitation(
                doc_id=doc_id,
                document=str(document["name"]),
                page=matches[0][1],
                verification="text_span",
            ),
        )
