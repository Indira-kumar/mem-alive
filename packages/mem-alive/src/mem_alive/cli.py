"""Local JSON command line interface for agents and shell workflows."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from .backend.sqlite_backend import SQLiteBackend
from .core.memory import Memory
from .documents.pageindex_documents import PageIndexDocuments
from .embedding.local_embedding_provider import LocalEmbeddingProvider
from .ingestion.ollama_extractor import OllamaMemoryExtractor
from .schema.memory_schema import Memory as MemoryRecord


def _default_db() -> Path:
    if configured := os.environ.get("MEM_ALIVE_DB"):
        return Path(configured).expanduser()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_home / "mem-alive/memory.db"


def _metadata(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")
    return parsed


def _record(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "namespace": record.namespace,
        "content": record.content,
        "memory_type": record.memory_type,
        "metadata": record.metadata,
        "superseded": record.superseded,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mem-alive")
    parser.add_argument("--db", type=Path, default=_default_db())
    parser.add_argument("--namespace", required=True, help="Hard scope for memories and documents")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    commands = parser.add_subparsers(dest="command", required=True)

    remember = commands.add_parser("remember")
    remember.add_argument("memory_type", choices=("semantic", "episodic", "procedural"))
    remember.add_argument("fact")
    remember.add_argument("--metadata", default="{}")

    recall = commands.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--type", choices=("semantic", "episodic", "procedural"))
    recall.add_argument("--limit", type=int, default=5)
    recall.add_argument("--metadata", default="{}")

    browse = commands.add_parser("list")
    browse.add_argument("--type", choices=("semantic", "episodic", "procedural"))
    browse.add_argument("--limit", type=int, default=100)
    browse.add_argument("--offset", type=int, default=0)
    browse.add_argument("--metadata", default="{}")

    get = commands.add_parser("get")
    get.add_argument("memory_id")
    history = commands.add_parser("history")
    history.add_argument("memory_id")
    update = commands.add_parser("update")
    update.add_argument("memory_id")
    update.add_argument("fact")
    update.add_argument("--metadata")
    forget = commands.add_parser("forget")
    forget.add_argument("memory_id")
    delete = commands.add_parser("delete")
    delete.add_argument("memory_id")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)

    ingest = commands.add_parser("ingest")
    ingest.add_argument("--file", type=Path, help="JSON message array; stdin when omitted")
    ingest.add_argument("--model", default="llama3.2")
    ingest.add_argument("--metadata", default="{}")
    ingest.add_argument("--store-source-text", action="store_true")

    index = commands.add_parser("index-pdf")
    index.add_argument("path", type=Path)
    index.add_argument("--model", default="llama3.2")
    ask = commands.add_parser("ask-doc")
    ask.add_argument("doc_id")
    ask.add_argument("question")
    ask.add_argument("--model", default="llama3.2")
    commands.add_parser("list-docs").add_argument("--model", default="llama3.2")
    delete_doc = commands.add_parser("forget-doc")
    delete_doc.add_argument("doc_id")
    delete_doc.add_argument("--model", default="llama3.2")
    return parser


async def _documents(args: argparse.Namespace, db_path: Path) -> object:
    import hashlib

    directory = db_path.parent / "documents" / hashlib.sha256(
        args.namespace.encode("utf-8")
    ).hexdigest()
    backend = {"api_key": "ollama", "base_url": f"{args.ollama_url.rstrip('/')}/v1"}
    documents = PageIndexDocuments.local(
        directory,
        index_model=f"openai/{args.model}",
        chat_model=f"openai/{args.model}",
        index_backend=backend,
        chat_backend=backend,
    )
    if args.command == "index-pdf":
        return {"doc_id": await documents.index_pdf(args.path)}
    if args.command == "ask-doc":
        answer = await documents.ask(args.doc_id, args.question)
        return {
            "answer": answer.text,
            "citations": [vars(citation) for citation in answer.citations],
        }
    if args.command == "list-docs":
        return await documents.list_documents()
    await documents.delete_document(args.doc_id)
    return {"deleted": args.doc_id}


async def _run(args: argparse.Namespace) -> object:
    db_path = args.db.expanduser()
    if args.command in ("index-pdf", "ask-doc", "list-docs", "forget-doc"):
        return await _documents(args, db_path)
    db = SQLiteBackend(db_path)
    if args.command == "backup":
        return {"backup": str(await db.backup(args.destination.expanduser()))}
    async with LocalEmbeddingProvider(
        url=args.ollama_url, model=args.embedding_model
    ) as embedder:
        memory = Memory(embedder, db)
        if args.command == "remember":
            return _record(
                await memory.remember(
                    args.memory_type, args.namespace, args.fact, _metadata(args.metadata)
                )
            )
        if args.command == "recall":
            hits = await memory.recall_hits(
                args.namespace, args.query, _metadata(args.metadata), args.limit, args.type
            )
            return [{**_record(hit.memory), "score": hit.score} for hit in hits]
        if args.command == "list":
            records = await memory.list(
                args.namespace, _metadata(args.metadata), args.type, args.limit, args.offset
            )
            return [_record(record) for record in records]
        if args.command == "get":
            record = await memory.get(args.namespace, args.memory_id)
            return _record(record) if record else None
        if args.command == "history":
            events = await memory.history(args.namespace, args.memory_id)
            return [
                {"action": event.action, "occurred_at": event.occurred_at.isoformat(),
                 "memory": _record(event.memory)}
                for event in events
            ]
        if args.command == "update":
            metadata = _metadata(args.metadata) if args.metadata is not None else None
            return _record(await memory.update(args.namespace, args.memory_id, args.fact, metadata))
        if args.command == "delete":
            await memory.delete(args.namespace, args.memory_id)
            return {"deleted": args.memory_id, "history_retained": True}
        if args.command == "forget":
            await memory.forget(args.namespace, args.memory_id)
            return {"forgotten": args.memory_id, "history_retained": False}
        source = args.file.read_text() if args.file else sys.stdin.read()
        messages = json.loads(source)
        if not isinstance(messages, list):
            raise ValueError("ingest input must be a JSON message array")
        if any(
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("ingest messages need user or assistant roles and text content")
        async with OllamaMemoryExtractor(model=args.model, url=args.ollama_url) as extractor:
            records = await memory.ingest(
                args.namespace,
                messages,
                extractor,
                _metadata(args.metadata),
                store_source_text=args.store_source_text,
            )
        return [_record(record) for record in records]


def main() -> None:
    os.umask(0o077)
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except (ValueError, KeyError, FileExistsError, httpx.HTTPError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
