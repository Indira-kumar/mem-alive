# mem-alive

Local, persistent memory for agents, with an async Python API, MCP tools, and an
optional HTTP adapter. Models run through Ollama or an OpenAI-compatible endpoint;
SQLite keeps memories, entity vectors, recent context, and revision history on disk.

This branch implements the architecture described in [Agent Memory EXPLAINED –
Complete Architecture](https://www.youtube.com/watch?v=aYfZN8t6AQs). See the
[design and acceptance record](https://github.com/Indira-kumar/mem-alive/blob/main/docs/local-agent-memory.md) for source references,
verification evidence, and the remaining v1 release gates. This is a release
candidate implementation, not a claim of full Mem0 product or SDK compatibility.

## Run as an MCP server

Install Python 3.11 or newer and Ollama, then prepare both local models:

```sh
ollama pull qwen3:8b
ollama pull embeddinggemma
pip install 'mem-alive[mcp]'
mem-alive-mcp --db /absolute/path/memory.sqlite3 --namespace my-project
```

Until this branch is released, install from this checkout instead:
`pip install './packages/mem-alive[mcp]'`.
The database's parent directory must already exist. Use the same embedding model
for the lifetime of a database; changing dimensions or the runtime's model name
is rejected. Model weights and Ollama are separate from the Python package.

Configure any client supporting MCP stdio, such as a coding agent, with:

```json
{
  "mcpServers": {
    "memory": {
      "command": "mem-alive-mcp",
      "args": [
        "--db", "/absolute/path/memory.sqlite3",
        "--namespace", "my-project"
      ]
    }
  }
}
```

Client configuration formats vary. The executable and arguments above are the
server contract; register them through your client's MCP configuration. Tools:
`add_memory`, `search_memories`, `get_memory`, `list_memories`, `update_memory`,
`delete_memory`, and `memory_history`. The configured namespace is fixed; a tool
caller cannot override it. Agent instructions must tell the agent when to save
and search: installing tools alone does not capture every turn automatically.

## Python

```python
import asyncio
from mem_alive.runtime import build_memory

async def main():
    async with build_memory("memory.sqlite3") as memory:
        await memory.add(
            [{"role": "user", "content": "This repository uses pytest."}],
            namespace="my-project",
            metadata={"user_id": "alice", "agent_id": "coding-agent"},
        )
        hits = await memory.search("How do I test this repository?", "my-project")
        print(hits)

asyncio.run(main())
```

`AgentMemory` accepts injected `EmbeddingProvider`, `InferenceProvider`, and
`SQLiteBackend` instances. `JSONInferenceProvider` supports Ollama and
OpenAI-compatible chat APIs; a compatible schema-capable model is required.
This lets a framework call memory at its own turn boundaries without a framework
dependency. MCP and HTTP are adapters over this same service.

## How memory works

Ingestion loads the scoped summary, the last ten messages, recent memories, and
memories related to the new turn. A model extracts facts, named entities, and an
updated summary. Exact text is deduplicated by SHA-256. Memories, entity links,
context, and audit changes commit in one transaction. Failures leave no partial
capture. `infer=False` stores raw message text without model inference or entity
extraction; embeddings are still required.

Ingestion is additive: a newly stated correction does not automatically erase an
older memory. Use `update` or `delete` when replacing information explicitly.
Updates regenerate embeddings, hashes, keyword tokens, and entity links.

Retrieval takes `max(4*K,60)` semantic candidates, then reranks those candidates
using lemmatized BM25 and entity specificity. Keyword or entity matches cannot
introduce additional memories. Results include scoring components. Namespace is
a hard boundary; metadata filters narrow the candidate set within it.

## HTTP and other integrations

Install `mem-alive[api]`, construct a service, and pass it to
`mem_alive.api.create_app(memory, namespace="my-project")`. Run that ASGI app with
your preferred server. The adapter owns the service lifetime and provides
`POST /memories`, `POST /memories/search`, scoped CRUD, and revision history.
It supplies no authentication: bind to loopback or add authentication in the host.
This is a mem-alive API, not a Mem0 wire-compatible replacement. Frameworks can
use the Python API, HTTP, or their MCP integration; bundled Mem0-specific SDK
adapters and its hosted dashboard are outside this implementation.

## Data and operational limits

All persistent state stays in the selected SQLite file (plus SQLite WAL files).
The two vector collections are separate logical tables with exact cosine search;
search scans the namespace and is intended for local collections, not a claim of
ANN-scale throughput. English dictionary lemmatization is local and requires no
model downloads. Other languages retain token search without morphological parity.

Delete removes the active memory and entity links. Audit history retains prior
content; it is not a secure-erasure API. Treat the database, summaries, and recent
messages as private data. A namespace is a partition, not authentication. Separate
untrusted users through separate service instances or an authenticated host.

The original `Memory` three-store API remains for existing callers. It retains
its earlier similarity-based supersession behavior; new applications should use
`AgentMemory`. The older evals document this API's limitations separately.

## Development and release

```sh
uv run --package mem-alive --all-extras --group dev pytest -q
uv run --package mem-alive --all-extras --group dev ruff check .
uv run --package mem-alive --all-extras --group dev ty check
uv run --package mem-alive --all-extras --group dev python -m evals.sqlite
uv run --package mem-alive --all-extras --group dev python -m evals.agent_memory
uv run --package mem-alive --all-extras --group dev python -m evals.hybrid
uv run --package mem-alive --all-extras --group dev python -m evals.mcp
uv run --package mem-alive --all-extras --group dev python -m evals.agent_memory --live
uv build --package mem-alive
```

Scripted evals isolate persistence, pipeline, ranking, and transport behavior. They
do not measure model quality. The live eval requires a running local model server
and installed weights and must pass before declaring v1 production ready.
Publishing remains an explicit maintainer action; see [releasing](https://github.com/Indira-kumar/mem-alive/blob/main/docs/releasing.md).
