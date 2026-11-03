"""Deterministic MCP/HTTP delivery eval against a real, reopened SQLite DB.

Run with ``uv run --package mem-alive --all-extras python -m evals.mcp``.
Scripted extraction and fixed vectors isolate transport/persistence correctness;
this eval makes no claim about live model quality or coding-agent task success.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from mem_alive.api import create_app
from mem_alive.backend.sqlite_backend import SQLiteBackend
from mem_alive.core.agent_memory import AgentMemory
from mem_alive.embedding.embedding_provider import EmbeddingProvider
from mem_alive.mcp import create_server


class FixedEmbedding(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class ScriptedInference:
    async def extract(
        self, messages: list[dict[str, str]], *, context: dict | None = None
    ) -> list[str]:
        return [message["content"] for message in messages if message["role"] == "user"]

    async def extract_entities(self, texts: list[str]) -> list[list[str]]:
        return [["pytest"] for _ in texts]

    async def summarize(self, previous_summary: str, messages: list[dict[str, str]]) -> str:
        return "Project tests use pytest."


def build(path: Path) -> AgentMemory:
    return AgentMemory(SQLiteBackend(path), FixedEmbedding(), ScriptedInference())


async def evaluate() -> dict:
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    checks = {}
    with TemporaryDirectory(prefix="mem-alive-surface-eval-") as directory:
        path = Path(directory) / "memories.sqlite3"
        service = build(path)
        server = create_server(service, "project-a")
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            response = await client.call_tool(
                "add_memory",
                {
                    "messages": [{"role": "user", "content": "Run tests using uv run pytest"}],
                },
            )
            checks["mcp_write"] = not response.isError and len(await service.list("project-a")) == 1
        # A fresh client and a new SQLite connection represent a later agent session.
        service = build(path)
        server = create_server(service, "project-a")
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            response = await client.call_tool("search_memories", {"query": "test command"})
            content = response.content[0]
            if not isinstance(content, TextContent):
                raise TypeError("MCP search returned non-text content")
            hits = json.loads(content.text)
            checks["mcp_reopen_recall"] = (
                not response.isError and hits["content"] == "Run tests using uv run pytest"
            )
            memory_id = hits["id"]
            response = await client.call_tool(
                "update_memory",
                {
                    "memory_id": memory_id,
                    "content": "Run tests using uv run pytest -q",
                },
            )
            updated = await service.get(memory_id, "project-a")
            checks["mcp_update"] = (
                not response.isError
                and updated is not None
                and updated["content"] == "Run tests using uv run pytest -q"
            )
            checks["mcp_audit"] = len(await service.history(memory_id, "project-a")) == 2
        foreign = build(path)
        server = create_server(foreign, "project-b")
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            response = await client.call_tool("search_memories", {"query": "test command"})
            checks["mcp_scope"] = not response.isError and response.content == []
        # The HTTP adapter reads the same local database, without a second memory engine.
        service = build(path)
        app = create_app(service, "project-a")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://eval"
            ) as client:
                response = await client.get(f"/memories/{memory_id}")
                checks["http_shared_persistence"] = (
                    response.status_code == 200
                    and response.json()["content"] == "Run tests using uv run pytest -q"
                )
                response = await client.post(
                    "/memories",
                    json={
                        "messages": [{"role": "user", "content": "secret"}],
                        "namespace": "project-b",
                    },
                )
                checks["http_scope_override_rejected"] = response.status_code == 422
                response = await client.delete(f"/memories/{memory_id}")
                checks["http_delete"] = response.status_code == 204
        service = build(path)
        try:
            checks["delete_survives_reopen"] = await service.get(memory_id, "project-a") is None
        finally:
            await service.aclose()
    return {"kind": "deterministic-surface-eval", "checks": checks, "passed": all(checks.values())}


def main() -> None:
    report = asyncio.run(evaluate())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
