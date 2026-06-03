"""Layer 0 — frontend-agnostic generation core for edge-lm.

This layer is pure of HTTP and asyncio (decision dec-20260603-903c8f4c): it
holds the warm ``(model, tokenizer)`` plus a small pool of reusable KV caches,
and turns ``(messages, params)`` into a stream of text deltas. The OpenAI HTTP
server and any future voice frontend are thin shells over this one core, never
the other way around.

Only a single worker thread is expected to drive a ``ModelSession`` (see
``edge_lm.serve.worker``), so the cache pool needs no locking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, cast

import mlx.core as mx
from mlx_vlm import PromptCacheState, stream_generate

from edge_lm.models.load import load


# ---------------------------------------------------------------------------
# Layer-0 data types — strong types parsed at the shell boundary, never raw
# protocol dicts leaking inward.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationParams:
    """Sampler knobs. Immutable: a request's settings can't be mutated mid-run."""

    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: tuple[str, ...] = ()


@dataclass
class GenerationChunk:
    """One streamed step: a text delta plus cumulative token bookkeeping.

    ``finish_reason`` is ``None`` for mid-stream chunks and one of
    ``"stop"`` / ``"length"`` on the terminal chunk.
    """

    text: str
    token: Optional[int] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0


# ---------------------------------------------------------------------------
# KV-cache pool — the staged "single/small cache" step toward the V2 LRU
# (dec-20260603-903c8f4c). mlx-vlm's PromptCacheState tracks a conversation's
# tokens + KV cache and is updated in place by stream_generate after each turn,
# so a follow-up whose prompt extends a prior conversation reuses that warm
# cache and only prefills the new tokens. Capacity 1 reduces this to plain V1.
# ---------------------------------------------------------------------------

_MIN_REUSE_PREFIX = 8     # below this overlap, treat the request as a fresh conversation
_REUSE_FRACTION = 0.8     # the new prompt must cover >=80% of a cached conversation


class ConversationCachePool:
    """Bounded MRU pool of per-conversation KV caches; longest-prefix reuse."""

    def __init__(self, capacity: int = 8):
        self._capacity = max(1, capacity)
        self._states: list[PromptCacheState] = []  # least- ... most-recently-used

    def acquire(self, prompt_ids: list[int]) -> PromptCacheState:
        state = self._best_match(prompt_ids)
        if state is None:
            state = self._new_state()
        self._touch(state)
        return state

    def _best_match(self, prompt_ids: list[int]) -> Optional[PromptCacheState]:
        # Reuse the cache whose conversation this prompt most fully continues.
        # The fractional gate accepts a follow-up turn (covers nearly all of the
        # cached tokens) but rejects a different conversation that merely shares a
        # short system-prompt prefix — which would otherwise clobber a live cache.
        viable: list[tuple[PromptCacheState, int]] = []
        for state in self._states:
            token_ids = state.token_ids
            if state.cache is None or token_ids is None:
                continue
            overlap = state.find_prefix_length(prompt_ids)
            if overlap >= _MIN_REUSE_PREFIX and overlap >= _REUSE_FRACTION * len(token_ids):
                viable.append((state, overlap))
        if not viable:
            return None
        return max(viable, key=lambda pair: pair[1])[0]

    def _new_state(self) -> PromptCacheState:
        if len(self._states) >= self._capacity:
            self._states.pop(0)  # evict least-recently-used
        state = PromptCacheState()
        self._states.append(state)
        return state

    def _touch(self, state: PromptCacheState) -> None:
        self._states.remove(state)
        self._states.append(state)


# ---------------------------------------------------------------------------
# ModelSession — the warm model + cache pool. One per process, driven by
# exactly one worker thread.
# ---------------------------------------------------------------------------

class ModelSession:
    def __init__(self, model, tokenizer, cache_capacity: int = 8):
        self._model = model
        self._tokenizer = tokenizer
        self._cache = ConversationCachePool(cache_capacity)

    @classmethod
    def from_pretrained(
        cls, model_id: str, size: Optional[str] = "m", cache_capacity: int = 8,
    ) -> "ModelSession":
        model, tokenizer = load(model_id, size=size)
        return cls(model, tokenizer, cache_capacity=cache_capacity)

    def render_prompt_ids(self, messages, tools=None) -> list[int]:
        template_kwargs = {"tools": tools} if tools else {}
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        return [int(token_id) for token_id in self._tokenizer.encode(prompt)]

    def generate(
        self, messages, params: GenerationParams, tools=None,
    ) -> Iterator[GenerationChunk]:
        """Stream text deltas for a chat turn, reusing a warm KV cache if one fits."""
        prompt_ids = self.render_prompt_ids(messages, tools=tools)
        prompt_token_count = len(prompt_ids)
        state = self._cache.acquire(prompt_ids)
        stops = tuple(s for s in params.stop if s)
        # Hold back the last (longest stop length - 1) chars before yielding, so a
        # stop sequence split across chunks (e.g. "E" then "ND" with stop "END")
        # is never partially emitted and then impossible to retract. 0 when there
        # are no stops, so plain streaming keeps emitting every char immediately.
        hold_back = max((len(s) for s in stops), default=1) - 1

        # stream_generate is annotated -> str | Generator[str], but with
        # input_ids set it yields GenerationResult records; cast so attribute
        # access type-checks.
        results = cast(Iterator[Any], stream_generate(
            self._model,
            self._tokenizer,
            "",
            input_ids=mx.array([prompt_ids], dtype=mx.int32),
            prompt_cache_state=state,
            **_sampler_kwargs(params),
        ))

        full_text = ""
        emitted = 0
        generation_tokens = 0
        finished = False
        for result in results:
            if finished:
                # Keep draining: stream_generate updates the reusable KV cache
                # only after its yield loop is exhausted (mlx_vlm dispatch.py).
                # Returning here would skip that update and lose prefix reuse.
                continue

            generation_tokens = result.generation_tokens or generation_tokens
            full_text += result.text

            stop_at = _first_stop_index(full_text, stops)
            if stop_at is not None:
                delta = full_text[emitted:stop_at]
                if delta:
                    yield GenerationChunk(
                        delta, result.token, None, prompt_token_count, generation_tokens,
                    )
                yield GenerationChunk(
                    "", None, "stop", prompt_token_count, generation_tokens,
                )
                return  # stop string: deliberate early abort; this turn isn't cached

            if result.finish_reason is not None:
                # Generation ended with no stop hit: flush the held-back tail.
                tail = full_text[emitted:]
                if tail:
                    yield GenerationChunk(
                        tail, result.token, None, prompt_token_count, generation_tokens,
                    )
                emitted = len(full_text)
                yield GenerationChunk(
                    "", None, result.finish_reason, prompt_token_count, generation_tokens,
                )
                finished = True
                continue

            # Mid-stream: emit only text that cannot be the start of a stop
            # sequence, holding back the last `hold_back` chars in case a stop
            # straddles the next chunk.
            safe_end = len(full_text) - hold_back
            if safe_end > emitted:
                yield GenerationChunk(
                    full_text[emitted:safe_end], result.token, None,
                    prompt_token_count, generation_tokens,
                )
                emitted = safe_end


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _sampler_kwargs(params: GenerationParams) -> dict:
    """Map our strong params onto the kwargs mlx_vlm.generate_step accepts."""
    kwargs: dict = {
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
    }
    if params.top_k and params.top_k > 0:
        kwargs["top_k"] = params.top_k
    if params.frequency_penalty:
        kwargs["frequency_penalty"] = params.frequency_penalty
    if params.presence_penalty:
        kwargs["presence_penalty"] = params.presence_penalty
    if params.seed is not None:
        kwargs["seed"] = params.seed
    return kwargs


def _first_stop_index(text: str, stops: tuple[str, ...]) -> Optional[int]:
    """Earliest index where any stop string begins, or None."""
    hits = [text.find(stop) for stop in stops]
    present = [index for index in hits if index != -1]
    return min(present) if present else None
