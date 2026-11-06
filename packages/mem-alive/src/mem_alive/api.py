"""Optional HTTP adapter for local integrations; this is not the Mem0 wire API."""

import sqlite3
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .core.agent_memory import AgentMemory


def create_app(memory: "AgentMemory", namespace: str) -> "FastAPI":
    """Create an application bound to one namespace and own the client lifetime.

    This adapter supplies no authentication. Bind it to loopback, or place it
    behind an authenticated application before exposing it to a network.
    """
    from importlib.metadata import version

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field

    from .backend.sqlite_backend import ConcurrentWriteError

    if not namespace.strip():
        raise ValueError("namespace must not be empty")

    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class AddRequest(Payload):
        messages: list[dict[str, str]]
        metadata: dict | None = None
        infer: bool = True

    class SearchRequest(Payload):
        query: str = Field(min_length=1)
        limit: int = Field(default=10, ge=1, le=1000)
        metadata: dict | None = None
        threshold: float = Field(default=0.3, ge=0, le=1)

    class UpdateRequest(Payload):
        content: str = Field(min_length=1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await memory.aclose()

    app = FastAPI(title="mem-alive", version=version("mem-alive"), lifespan=lifespan)

    @app.exception_handler(ConcurrentWriteError)
    async def concurrent_write(request, exc):
        return JSONResponse(
            status_code=409, content={"detail": "Memory changed; retry the operation"}
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_unavailable(request, exc):
        return JSONResponse(status_code=503, content={"detail": "Memory storage unavailable"})

    @app.post("/memories")
    async def add_memories(payload: AddRequest) -> list[dict]:
        try:
            return await memory.add(
                payload.messages,
                namespace=namespace,
                metadata=payload.metadata,
                infer=payload.infer,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/memories/search")
    async def search_memories(payload: SearchRequest) -> list[dict]:
        try:
            return await memory.search(
                payload.query,
                namespace=namespace,
                limit=payload.limit,
                metadata=payload.metadata,
                threshold=payload.threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/memories")
    async def list_memories(limit: int = 100, offset: int = 0) -> list[dict]:
        try:
            return await memory.list(namespace=namespace, limit=limit, offset=offset)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/memories/{memory_id}")
    async def get_memory(memory_id: str) -> dict:
        result = await memory.get(memory_id, namespace=namespace)
        if result is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return result

    @app.patch("/memories/{memory_id}")
    async def update_memory(memory_id: str, payload: UpdateRequest) -> dict:
        try:
            return await memory.update(memory_id, payload.content, namespace=namespace)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Memory not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/memories/{memory_id}", status_code=204)
    async def delete_memory(memory_id: str) -> None:
        await memory.delete(memory_id, namespace=namespace)

    @app.get("/memories/{memory_id}/history")
    async def memory_history(memory_id: str) -> list[dict]:
        return await memory.history(memory_id, namespace=namespace)

    return app
