import json

import httpx
import pytest
from mem_alive.inference import JSONInferenceProvider


def ollama_body(content):
    return {"message": {"content": content}, "done": True, "done_reason": "stop"}


async def test_ollama_extraction_uses_context_and_structured_requests():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        assert request.url == "http://models.test/api/chat"
        return httpx.Response(200, json=ollama_body('{"facts":["Use pytest"]}'))

    context = {
        "summary": "Alice maintains billing",
        "recent_messages": [],
        "relevant_memories": ["Use pytest"],
        "recent_memories": [],
        "current_date": "2026-11-04",
        "conversation_date": "2026-11-03",
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = JSONInferenceProvider("qwen3:8b", base_url="http://models.test", client=client)
        assert await provider.extract(
            [{"role": "user", "content": "Use pytest"}], context=context
        ) == ["Use pytest"]
        assert requests[0]["format"]["required"] == ["facts"]
        assert requests[0]["options"]["temperature"] == 0
        assert requests[0]["stream"] is False
        supplied = json.loads(requests[0]["messages"][1]["content"])
        assert supplied["context"] == context
        await provider.aclose()
        assert not client.is_closed


async def test_openai_compatible_endpoint_auth_and_schema():
    def handle(request):
        assert request.url == "https://models.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-token"
        payload = json.loads(request.content)
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["model"] == "model-id"
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"facts": []}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = JSONInferenceProvider(
            "model-id",
            provider="openai",
            base_url="https://models.test/v1/",
            api_key="test-token",
            client=client,
        )
        assert await provider.extract([{"role": "user", "content": "Hello"}]) == []


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '```json\n{"facts": []}\n```',
        "[]",
        '{"facts": [], "surprise": true}',
        '{"facts": "fact"}',
        '{"facts": [42]}',
        '{"facts": [" "]}',
        '{"facts": [], "facts": ["x"]}',
        json.dumps({"facts": ["fact"] * 101}),
    ],
)
async def test_malformed_extractions_fail_without_fallback(content):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=ollama_body(content))
        )
    ) as client:
        provider = JSONInferenceProvider("model", client=client)
        with pytest.raises(ValueError):
            await provider.extract([{"role": "user", "content": "Remember this"}])


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": {}},
        {"done": False, "message": {"content": '{"facts": []}'}},
        {"done": True, "done_reason": "length", "message": {"content": '{"facts": []}'}},
        {"done": True, "message": {"content": None}},
    ],
)
async def test_incomplete_model_envelopes_fail(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(ValueError):
            await JSONInferenceProvider("model", client=client).extract(
                [{"role": "user", "content": "Remember this"}]
            )


async def test_http_errors_propagate():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "unavailable"})
        )
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await JSONInferenceProvider("model", client=client).extract(
                [{"role": "user", "content": "Remember this"}]
            )


@pytest.mark.parametrize(
    "messages", [[], [{"role": "developer", "content": "x"}], [{"role": "user", "content": ""}]]
)
async def test_invalid_inputs_do_not_call_model(messages):
    def unexpected(request):
        pytest.fail("Invalid inputs must not reach the model")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        provider = JSONInferenceProvider("model", client=client)
        with pytest.raises(ValueError):
            await provider.extract(messages)
        assert await provider.extract_entities([]) == []


async def test_entities_are_aligned_and_deduplicated():
    def handle(request):
        data = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert data == {"texts": ["Alice uses Redis", "No named subject"]}
        return httpx.Response(200, json=ollama_body('{"entities":[["Alice","Redis","Alice"],[]]}'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = JSONInferenceProvider("model", client=client)
        assert await provider.extract_entities(["Alice uses Redis", "No named subject"]) == [
            ["Alice", "Redis"],
            [],
        ]


@pytest.mark.parametrize("entities", [[], [["a"], ["b"]], ["Alice"], [[""]], [[42]], [["a"] * 21]])
async def test_invalid_entities_fail(entities):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=ollama_body(json.dumps({"entities": entities}))
            )
        )
    ) as client:
        with pytest.raises(ValueError):
            await JSONInferenceProvider("model", client=client).extract_entities(["one text"])


@pytest.mark.parametrize("summary", ["Alice maintains billing", ""])
async def test_summary_generation_uses_previous_summary_and_new_messages(summary):
    messages = [{"role": "user", "content": "I maintain billing"}]

    def handle(request):
        data = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert data == {"previous_summary": "Alice writes Python", "messages": messages}
        return httpx.Response(200, json=ollama_body(json.dumps({"summary": summary})))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = JSONInferenceProvider("model", client=client)
        assert await provider.summarize("Alice writes Python", messages) == summary


@pytest.mark.parametrize(
    "result", [{"summary": 42}, {"summary": "a", "extra": 1}, {"summary": "x" * 16001}]
)
async def test_invalid_summaries_fail(result):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=ollama_body(json.dumps(result)))
        )
    ) as client:
        with pytest.raises(ValueError):
            await JSONInferenceProvider("model", client=client).summarize(
                "", [{"role": "user", "content": "hello"}]
            )
