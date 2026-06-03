"""Effect boundary — load the model once, start the worker, serve over HTTP.

    uv run --extra serve python -m edge_lm.serve --model TheStageAI/gemma-4-E4B-it --size l

All flags also read from EDGE_LM_* environment variables.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from edge_lm.serve.core import ModelSession
from edge_lm.serve.openai import create_app


def _parse_args() -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python -m edge_lm.serve")
    parser.add_argument("--model", default=env("EDGE_LM_MODEL", "TheStageAI/gemma-4-E2B-it"))
    parser.add_argument("--size", default=env("EDGE_LM_SIZE", "m"))
    parser.add_argument("--host", default=env("EDGE_LM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("EDGE_LM_PORT", "8000")))
    parser.add_argument("--max-tokens", type=int, default=int(env("EDGE_LM_MAX_TOKENS", "512")))
    parser.add_argument(
        "--cache-capacity", type=int, default=int(env("EDGE_LM_CACHE_CAPACITY", "8")),
        help="How many conversations keep a warm KV cache",
    )
    parser.add_argument(
        "--served-model-name", default=env("EDGE_LM_SERVED_MODEL_NAME"),
        help="Name reported to clients (defaults to --model)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    served = args.served_model_name or args.model

    print(f"Loading {args.model} (size={args.size})...")
    session = ModelSession.from_pretrained(
        args.model, size=args.size, cache_capacity=args.cache_capacity,
    )
    app = create_app(session, served_model_name=served, default_max_tokens=args.max_tokens)

    print(f"edge-lm serving '{served}' at http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
