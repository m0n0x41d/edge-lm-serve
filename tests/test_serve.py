"""Contract tests for the OpenAI-compatible serving layer.

These exercise the protocol shell and the generation core's stop / usage logic
WITHOUT loading a model — a stub session stands in for the real one, so the
suite runs in milliseconds. The full model-backed acceptance (a real completion
from Gemma 4) is a separate smoke run; see README "Serving".

Runnable two ways:
    uv run python tests/test_serve.py      # standalone, exits non-zero on failure
    uv run pytest tests/test_serve.py      # if pytest is installed
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import edge_lm.serve.core as core
from edge_lm.serve.core import GenerationChunk, GenerationParams, ModelSession
from edge_lm.serve.openai import create_app


# ---------------------------------------------------------------------------
# Test doubles — implement the Session protocol (generate -> chunk iterator)
# ---------------------------------------------------------------------------

class StubSession:
    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error
        self.calls: list = []

    def generate(self, messages, params, tools=None):
        self.calls.append((messages, params, tools))
        if self._error is not None:
            raise self._error
        yield from self._chunks


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw) -> str:
        return "PROMPT"

    def encode(self, text: str) -> list:
        return [10, 11, 12, 13]


def _sse_payloads(raw_lines) -> tuple[list[dict], bool]:
    datas = [line[len("data: "):] for line in raw_lines if line.startswith("data: ")]
    saw_done = "[DONE]" in datas
    payloads = [json.loads(d) for d in datas if d != "[DONE]"]
    return payloads, saw_done


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------

def test_non_stream_completion_shape():
    chunks = [
        GenerationChunk("Hello", token=1, prompt_tokens=5, generation_tokens=1),
        GenerationChunk(" world", token=2, prompt_tokens=5, generation_tokens=2),
        GenerationChunk("", finish_reason="stop", prompt_tokens=5, generation_tokens=2),
    ]
    with TestClient(create_app(StubSession(chunks), "test-model")) as client:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_stream_completion_sse_shape():
    chunks = [
        GenerationChunk("Hi", prompt_tokens=3, generation_tokens=1),
        GenerationChunk("!", prompt_tokens=3, generation_tokens=2),
        GenerationChunk("", finish_reason="stop", prompt_tokens=3, generation_tokens=2),
    ]
    with TestClient(create_app(StubSession(chunks), "test-model")) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }) as resp:
            assert resp.status_code == 200
            lines = list(resp.iter_lines())

    payloads, saw_done = _sse_payloads(lines)
    assert saw_done, "stream must terminate with data: [DONE]"
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    assert payloads[0]["choices"][0]["delta"].get("role") == "assistant"
    content = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads if p["choices"]
    )
    assert content == "Hi!"
    assert any(
        p["choices"] and p["choices"][0]["finish_reason"] == "stop" for p in payloads
    )
    usage_chunks = [p for p in payloads if p.get("usage")]
    assert usage_chunks and usage_chunks[-1]["usage"]["total_tokens"] == 5


def test_models_and_health():
    with TestClient(create_app(StubSession([]), "my-model")) as client:
        assert client.get("/health").json() == {"status": "ok", "model": "my-model"}
        models = client.get("/v1/models").json()
        assert models["object"] == "list"
        assert models["data"][0]["id"] == "my-model"


def test_error_surfaces_as_500():
    with TestClient(create_app(StubSession([], error=RuntimeError("boom")), "m")) as client:
        resp = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 500
    assert "boom" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Core: stop truncation + usage, with stream_generate stubbed out
# ---------------------------------------------------------------------------

def test_core_stop_truncation_and_usage():
    def fake_stream_generate(model, tokenizer, prompt, input_ids=None,
                             prompt_cache_state=None, **kw):
        steps = [(1, "Answer: 42"), (2, " END"), (3, " should not appear")]
        for token, text in steps:
            yield SimpleNamespace(
                text=text, token=token, finish_reason=None,
                prompt_tokens=4, generation_tokens=token,
            )

    original = core.stream_generate
    core.stream_generate = fake_stream_generate
    try:
        session = ModelSession(model=None, tokenizer=StubTokenizer())
        chunks = list(session.generate(
            [{"role": "user", "content": "q"}],
            GenerationParams(stop=("END",)),
        ))
    finally:
        core.stream_generate = original

    text = "".join(c.text for c in chunks)
    assert "END" not in text
    assert "should not appear" not in text
    assert text == "Answer: 42 "
    assert chunks[-1].finish_reason == "stop"
    assert chunks[0].prompt_tokens == 4  # len(StubTokenizer.encode(...))


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [
        test_non_stream_completion_shape,
        test_stream_completion_sse_shape,
        test_models_and_health,
        test_error_surfaces_as_500,
        test_core_stop_truncation_and_usage,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # report, don't hide
            failed += 1
            print(f"FAIL  {test.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
