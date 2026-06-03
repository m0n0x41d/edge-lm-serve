"""edge-lm serving layer.

Layered (decision dec-20260603-903c8f4c):

- ``core``   — Layer 0: frontend-agnostic generation (no HTTP / asyncio). The
  reusable unit a future voice frontend drives directly.
- ``openai`` — Layer 1/2: OpenAI-compatible HTTP shell that serializes
  generation on the event-loop thread (requires the ``serve`` extra).

This package init imports only the core, so it stays usable without FastAPI.
Import ``create_app`` from ``edge_lm.serve.openai`` for the HTTP server.
"""

from edge_lm.serve.core import GenerationChunk, GenerationParams, ModelSession

__all__ = ["GenerationChunk", "GenerationParams", "ModelSession"]
