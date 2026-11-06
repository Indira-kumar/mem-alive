"""Exercise adapters through their actual HTTP and MCP protocol boundaries."""

from unittest.mock import AsyncMock

import httpx
import pytest
from mem_alive.api import create_app
from mem_alive.mcp import create_server


@pytest.fixture
def memory():
    service = AsyncMock()
    service.add.return_value = [{"id": "one", "content": "Use pytest"}]
    service.search.return_value = [{"id": "one", "content": "Use pytest", "score": 0.9}]
    service.get.return_value = {"id": "one", "content": "Use pytest"}
    service.list.return_value = [{"id": "one", "content": "Use pytest"}]
    service.history.return_value = [{"action": "ADD", "memory_id": "one"}]
    service.update.return_value = {"id": "one", "content": "Use uv run pytest"}
    return service


async def test_http_crud_scope_validation_and_lifetime(memory):
    pytest.importorskip("fastapi")
    app = create_app(memory, "repo-a")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/memories", json={"messages": [{"role": "user", "content": "Use pytest"}]}
            )
            assert response.status_code == 200
            assert response.json()[0]["id"] == "one"
            memory.add.assert_awaited_once_with(
                [{"role": "user", "content": "Use pytest"}],
                namespace="repo-a",
                metadata=None,
                infer=True,
            )
            response = await client.post("/memories", json={"messages": [], "namespace": "repo-b"})
            assert response.status_code == 422
            assert memory.add.await_count == 1
            response = await client.post("/memories/search", json={"query": "tests"})
            assert response.status_code == 200
            assert response.json()[0]["score"] == 0.9
            memory.search.assert_awaited_once_with(
                "tests", namespace="repo-a", limit=10, metadata=None, threshold=0.3
            )
            assert (
                await client.post("/memories/search", json={"query": "tests", "limit": -1})
            ).status_code == 422
            assert (await client.get("/memories")).status_code == 200
            memory.list.assert_awaited_once_with(namespace="repo-a", limit=100, offset=0)
            assert (await client.get("/memories/one")).json()["id"] == "one"
            memory.get.assert_awaited_once_with("one", namespace="repo-a")
            assert (await client.get("/memories/one/history")).status_code == 200
            memory.history.assert_awaited_once_with("one", namespace="repo-a")
            response = await client.patch("/memories/one", json={"content": "Use uv run pytest"})
            assert response.status_code == 200
            memory.update.assert_awaited_once_with("one", "Use uv run pytest", namespace="repo-a")
            assert (await client.delete("/memories/one")).status_code == 204
            memory.delete.assert_awaited_once_with("one", namespace="repo-a")
    memory.aclose.assert_awaited_once()


async def test_http_missing_memory(memory):
    pytest.importorskip("fastapi")
    memory.get.return_value = None
    memory.update.side_effect = KeyError("missing")
    app = create_app(memory, "repo-a")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/memories/missing")).status_code == 404
        assert (await client.patch("/memories/missing", json={"content": "new"})).status_code == 404


async def test_mcp_protocol_tools_and_bound_scope(memory):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    server = create_server(memory, "repo-a")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == {
            "add_memory",
            "search_memories",
            "get_memory",
            "list_memories",
            "update_memory",
            "delete_memory",
            "memory_history",
        }
        assert all("namespace" not in tool.inputSchema.get("properties", {}) for tool in tools)
        response = await client.call_tool(
            "add_memory",
            {
                "messages": [{"role": "user", "content": "Use pytest"}],
                "infer": False,
            },
        )
        assert not response.isError
        memory.add.assert_awaited_once_with(
            [{"role": "user", "content": "Use pytest"}],
            namespace="repo-a",
            metadata=None,
            infer=False,
        )
        response = await client.call_tool("search_memories", {"query": "tests"})
        assert not response.isError
        memory.search.assert_awaited_once_with(
            "tests", namespace="repo-a", limit=10, metadata=None, threshold=0.3
        )
        for tool, arguments, method in [
            ("get_memory", {"memory_id": "one"}, memory.get),
            ("memory_history", {"memory_id": "one"}, memory.history),
            ("delete_memory", {"memory_id": "one"}, memory.delete),
        ]:
            response = await client.call_tool(tool, arguments)
            assert not response.isError
            method.assert_awaited_once_with("one", namespace="repo-a")
        assert not (await client.call_tool("list_memories", {})).isError
        memory.list.assert_awaited_once_with(namespace="repo-a", limit=100, offset=0, metadata=None)
        assert not (
            await client.call_tool(
                "update_memory", {"memory_id": "one", "content": "Use uv run pytest"}
            )
        ).isError
        memory.update.assert_awaited_once_with("one", "Use uv run pytest", namespace="repo-a")
    memory.aclose.assert_awaited_once()


def test_empty_namespace_rejected(memory):
    pytest.importorskip("mcp")
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError, match="namespace"):
        create_server(memory, " ")
    with pytest.raises(ValueError, match="namespace"):
        create_app(memory, " ")


async def test_http_conflict_and_storage_failure_hide_internal_details(memory):
    pytest.importorskip("fastapi")
    import sqlite3

    from mem_alive.backend.sqlite_backend import ConcurrentWriteError

    app = create_app(memory, "repo-a")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        memory.update.side_effect = ConcurrentWriteError("private memory payload")
        response = await client.patch("/memories/one", json={"content": "new"})
        assert response.status_code == 409
        assert "private" not in response.text
        memory.get.side_effect = sqlite3.OperationalError("/private/database/path")
        response = await client.get("/memories/one")
        assert response.status_code == 503
        assert "private" not in response.text


async def test_http_invalid_search_returns_validation_error(memory):
    pytest.importorskip("fastapi")
    app = create_app(memory, "repo-a")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/memories/search", json={"query": "q", "threshold": -0.5})
        assert response.status_code == 422
        memory.search.assert_not_awaited()
        memory.search.side_effect = ValueError("query must be nonempty text")
        response = await client.post("/memories/search", json={"query": " "})
        assert response.status_code == 422
