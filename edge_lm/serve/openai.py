"""Layer 2 — OpenAI-compatible HTTP protocol shell.

Thin by design (decision dec-20260603-903c8f4c): it parses OpenAI requests into
core ``GenerationParams``, drives the ``ModelSession``, and renders responses
(streaming SSE or whole-body). It holds no model state of its own.

Concurrency note (rollback of dec-20260603-903c8f4c's worker-thread invariant —
see the linked h-note): mlx-vlm's ``generation_stream`` is thread-local to the
import thread, so generation must run on the event-loop thread, not a worker
thread. Requests therefore serialize on a single ``asyncio.Lock`` and the
generator yields control between tokens so streaming stays responsive. The
reusable unit for a future voice frontend remains ``core.ModelSession``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Iterator, Optional, Protocol, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from edge_lm.serve.core import GenerationChunk, GenerationParams


# ---------------------------------------------------------------------------
# The shell depends on this interface, not the concrete ModelSession — so a
# test stub (or a different backend later) drops in without touching the routes.
# ---------------------------------------------------------------------------

class Session(Protocol):
    def generate(
        self, messages, params: GenerationParams, tools=None,
    ) -> Iterator[GenerationChunk]: ...


# ---------------------------------------------------------------------------
# OpenAI request schema (only the fields we honor; extras are ignored).
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, list, None] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Union[str, list[str], None] = None
    stream: bool = False
    stream_options: Optional[dict] = None
    tools: Optional[list] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    session: Session, served_model_name: str, default_max_tokens: int = 512,
) -> FastAPI:
    app = FastAPI(title="edge-lm OpenAI-compatible server")
    lock = asyncio.Lock()  # one generation at a time on the event-loop thread

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": served_model_name}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "edge-lm",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        messages = [_message_dict(m) for m in req.messages]
        params = _params_from_request(req, default_max_tokens)
        if req.stream:
            return _stream_response(
                session, lock, request, messages, params, req, served_model_name,
            )
        return await _whole_response(
            session, lock, request, messages, params, req, served_model_name,
        )

    return app


# ---------------------------------------------------------------------------
# Generation driver. Generation runs on this (event-loop) thread, serialized by
# the lock; sleep(0) between chunks lets the loop flush the SSE write and notice
# client disconnects.
# ---------------------------------------------------------------------------

async def _drive(
    session: Session, lock: asyncio.Lock, request: Request,
    messages, params: GenerationParams, tools,
) -> AsyncIterator[GenerationChunk]:
    async with lock:
        generator = session.generate(messages, params, tools=tools)
        try:
            for chunk in generator:
                if await request.is_disconnected():
                    return
                yield chunk
                await asyncio.sleep(0)
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()


# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------

async def _whole_response(
    session, lock, request, messages, params, req, model,
) -> JSONResponse:
    parts: list[str] = []
    finish_reason = "stop"
    prompt_tokens = completion_tokens = 0
    try:
        async for chunk in _drive(session, lock, request, messages, params, req.tools):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            prompt_tokens = chunk.prompt_tokens or prompt_tokens
            completion_tokens = chunk.generation_tokens or completion_tokens
    except Exception as exc:  # never a silent 200 with no body
        return _error_response(exc)

    body = {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(parts)},
            "finish_reason": finish_reason,
        }],
        "usage": _usage(prompt_tokens, completion_tokens),
    }
    return JSONResponse(body)


def _stream_response(
    session, lock, request, messages, params, req, model,
) -> StreamingResponse:
    completion_id = _completion_id()
    created = int(time.time())
    include_usage = bool((req.stream_options or {}).get("include_usage"))

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(_chunk(completion_id, created, model, delta={"role": "assistant"}))
        finish_reason = "stop"
        prompt_tokens = completion_tokens = 0
        try:
            async for chunk in _drive(
                session, lock, request, messages, params, req.tools,
            ):
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                prompt_tokens = chunk.prompt_tokens or prompt_tokens
                completion_tokens = chunk.generation_tokens or completion_tokens
                if chunk.text:
                    yield _sse(_chunk(
                        completion_id, created, model, delta={"content": chunk.text},
                    ))
        except Exception as exc:
            yield _sse(_error_body(exc))
            yield "data: [DONE]\n\n"
            return

        yield _sse(_chunk(
            completion_id, created, model, delta={}, finish_reason=finish_reason,
        ))
        if include_usage:
            yield _sse(_usage_chunk(
                completion_id, created, model, prompt_tokens, completion_tokens,
            ))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Pure builders
# ---------------------------------------------------------------------------

def _message_dict(message: ChatMessage) -> dict:
    return {"role": message.role, "content": _coerce_content(message.content)}


def _coerce_content(content: Union[str, list, None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text_parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(text_parts)


def _params_from_request(
    req: ChatCompletionRequest, default_max_tokens: int,
) -> GenerationParams:
    max_tokens = req.max_completion_tokens or req.max_tokens or default_max_tokens
    if isinstance(req.stop, str):
        stop: tuple[str, ...] = (req.stop,)
    else:
        stop = tuple(req.stop or ())
    return GenerationParams(
        max_tokens=max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        seed=req.seed,
        stop=stop,
    )


def _chunk(
    completion_id: str, created: int, model: str,
    delta: dict, finish_reason: Optional[str] = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _usage_chunk(
    completion_id: str, created: int, model: str,
    prompt_tokens: int, completion_tokens: int,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _error_body(exc: Exception) -> dict:
    return {"error": {"message": str(exc), "type": exc.__class__.__name__, "code": None}}


def _error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(_error_body(exc), status_code=500)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex
