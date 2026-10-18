"""Live Ollama conversation extraction eval with positive and negative cases."""

import asyncio
from datetime import datetime, timezone

from mem_alive import OllamaMemoryExtractor


async def evaluate() -> None:
    cases = [
        (
            "preference",
            [{"role": "user", "content": "I prefer dark mode in coding tools."}],
            "semantic",
            ("dark", "mode"),
        ),
        (
            "event",
            [{"role": "user", "content": "Yesterday I merged PR 42 for mem-alive."}],
            "episodic",
            ("42",),
        ),
        (
            "workflow",
            [{"role": "user", "content": "When deploying, run tests and then build the package."}],
            "procedural",
            ("test", "build"),
        ),
        (
            "correction",
            [
                {"role": "assistant", "content": "You prefer tea."},
                {"role": "user", "content": "Actually, I prefer coffee."},
            ],
            "semantic",
            ("coffee",),
        ),
        (
            "assistant suggestion",
            [
                {"role": "user", "content": "What database should I use?"},
                {"role": "assistant", "content": "SQLite is a good option."},
            ],
            None,
            (),
        ),
        (
            "small talk",
            [
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello!"},
            ],
            None,
            (),
        ),
    ]
    passed = 0
    async with OllamaMemoryExtractor(timeout=90) as extractor:
        for name, messages, memory_type, terms in cases:
            extracted = await asyncio.wait_for(extractor.extract(messages), timeout=100)
            if memory_type is None:
                ok = extracted == []
            else:
                ok = any(
                    item.memory_type == memory_type
                    and all(term in item.content.lower() for term in terms)
                    and (name != "correction" or "tea" not in item.content.lower())
                    for item in extracted
                )
            passed += ok
            print(f"{name}: {'PASS' if ok else 'FAIL'} {extracted}", flush=True)
    print(
        f"model=ollama/llama3.2 date={datetime.now(timezone.utc).date()} "
        f"extraction={passed}/{len(cases)}"
    )
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
