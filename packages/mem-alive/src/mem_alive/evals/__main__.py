import argparse
import asyncio
import json
from dataclasses import asdict

from ..backend.in_memory_backend import InMemoryBackend
from ..core.memory import Memory
from ..embedding.local_embedding_provider import LocalEmbeddingProvider
from .cases import default_cases
from .ollama_chat_provider import OllamaChatProvider
from .rag_agent import RagAgent
from .runner import EvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mem-alive evals against local Ollama models")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--chat-model", default="qwen3:14b")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    backend = InMemoryBackend()
    async with (
        LocalEmbeddingProvider(
            base_url=args.base_url,
            model=args.embedding_model,
            timeout=args.timeout,
        ) as embeddings,
        OllamaChatProvider(
            base_url=args.base_url,
            model=args.chat_model,
            timeout=args.timeout,
        ) as chat,
    ):
        memory = Memory(embedding_provider=embeddings, db=backend)
        agent = RagAgent(memory=memory, chat_provider=chat)
        report = await EvalRunner(memory=memory, agent=agent).run(default_cases())

    print(json.dumps(asdict(report), indent=2))
    return 0 if report.passed else 1


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
