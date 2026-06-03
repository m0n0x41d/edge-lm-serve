# edge-lm-serve

![Gemma E2B compression flow: 9.26 GB BF16 compressed to 1.44 GB — 6.4× smaller](https://cdn.thestage.ai/production/cms_file_upload/1780406294-645b80f9-cebe-4ef2-bc04-f524afb4f244/Tokens%20per%20Second%20CuDNN%20%282%29.png)

**A private, OpenAI-compatible local LLM server for Apple Silicon.**

`edge-lm-serve` keeps a compressed Gemma 4 model resident (warm) in memory on your Mac and serves it over an OpenAI-compatible HTTP API — point any OpenAI client, SDK, or tool at `localhost` and get private, on-device completions with no cloud round-trip.

It is built on [**edge-lm**](https://github.com/TheStageAI/edge-lm) by [TheStageAI](https://thestage.ai), which provides the ~7× smaller Gemma 4 checkpoints and the MLX inference core (the `edge_lm` package). This project adds the warm-model serving layer on top — and is growing toward an on-device Mac assistant built on the same core. See [Acknowledgments](#acknowledgments).

## Quick start

```bash
git clone https://github.com/m0n0x41d/edge-lm-serve.git
cd edge-lm-serve

python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]"        # inference core + fastapi/uvicorn
```

Start the server (downloads `TheStageAI/gemma-4-E2B-it` on first run):

```bash
python -m edge_lm.serve          # OpenAI-compatible API at http://127.0.0.1:8000/v1
# python -m edge_lm.serve --model TheStageAI/gemma-4-E4B-it --size l --port 8000
```

Call it with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="TheStageAI/gemma-4-E2B-it",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

…or `curl`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TheStageAI/gemma-4-E2B-it", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

## How serving works

The model is loaded once at startup and stays resident. Requests serialize on a single lock (one generation at a time on the Metal GPU) while streaming stays responsive, and a small pool of per-conversation KV caches lets follow-up turns skip re-prefill — **≈3.6× faster time-to-first-token on a cache hit** (measured on an Apple M3 Pro, E2B).

- **Endpoints:** `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`, `GET /health`.
- **Config** via flags or `EDGE_LM_*` env vars: `--model`, `--size`, `--host`, `--port`, `--max-tokens`, `--cache-capacity`, `--served-model-name`.

## Models

| Model | M size (default) | L size | Compression |
|---|---|---|---|
| [`TheStageAI/gemma-4-E2B-it`](https://huggingface.co/TheStageAI/gemma-4-E2B-it) | **1.44 GB** | 1.72 GB | up to 6.4× |
| [`TheStageAI/gemma-4-E4B-it`](https://huggingface.co/TheStageAI/gemma-4-E4B-it) | **2.72 GB** | 3.28 GB | up to 5.6× |

Weights download automatically from HuggingFace on first run. Each model ships two operating points — `l` (more quality, larger artifact) and `m` (the smaller headline compression target, default). These compressed checkpoints are TheStageAI's work — read the write-up: [*7× size reduction for Gemma 4 Edge models — Compressing PLE architectures*](https://app.thestage.ai/blog/7x-size-reduction-for-Gemma4-Edge-models?id=14).

## Python API (direct, no server)

```python
from edge_lm import load
from mlx_vlm import stream_generate

model, tokenizer = load()  # TheStageAI/gemma-4-E2B-it, size "m" by default
# model, tokenizer = load("TheStageAI/gemma-4-E4B-it", size="l")  # larger, higher quality

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a haiku about the moon."}],
    tokenize=False, add_generation_prompt=True,
)
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=128):
    print(chunk.text, end="", flush=True)
```

More examples:

```bash
python examples/test_vision.py --image photo.jpg --prompt "Describe this image"
python examples/test_audio.py  --audio recording.wav --prompt "Transcribe this speech"
python examples/chat.py --tools                      # interactive chat with tool use
```

## Benchmarks

The quality and performance numbers below are from **edge-lm (TheStageAI)** for the compressed checkpoints; `Ours` / `TheStage (ours)` refers to TheStageAI's compressed model.

### Quality

Every model — the compressed TheStage checkpoints and the GGUF baselines alike — is dequantized to a standard BF16 checkpoint and served through vLLM, so the backend is equalized across the table. Reported: **MMLU-Pro** (general knowledge), **IFEval** (instruction following), and **τ²-Bench / Tau2** (multi-step tool use). For Tau2 the Gemma checkpoint under test acts as the agent while a fixed `Qwen3-235B-A22B-2507` simulates the user.

`Ours L` keeps more quality at a larger artifact size; `Ours M` is the smaller headline compression target.

**Gemma 4 E2B**

| Model | Compression | MMLU-Pro | IFEval | Tau2 (avg of 3) |
|---|---|---|---|---|
| BF16 | 1.00× | 61.85 | 74.68 | 30.67 |
| **Ours L** | 5.62× | **54.48** | **74.86** | 22.20 |
| **Ours M** | **6.40×** | 49.85 | 71.53 | **23.45** |
| Unsloth Q3-K-S | 3.81× | 48.20 | 64.51 | 18.69 |
| Unsloth UD-Q2-K-XL | 3.87× | 43.17 | 66.54 | 20.23 |

**Gemma 4 E4B**

| Model | Compression | MMLU-Pro | IFEval | Tau2 |
|---|---|---|---|---|
| BF16 | 1.00× | 70.49 | 81.33 | 37.19 |
| **Ours L** | 4.64× | **67.41** | **81.52** | **33.25** |
| **Ours M** | **5.60×** | 63.54 | 80.78 | 29.04 |
| Unsloth Q3-K-S | 3.90× | 63.66 | 77.08 | 30.47 |
| Unsloth UD-Q2-K-XL | 4.01× | 58.69 | 79.67 | 22.91 |

Bold metric values mark the best result among the compressed checkpoints in each column. Tau2 computed with `Qwen3-235B-A22B-2507` as the user simulator.

Reproduce the quality benchmarks:

```bash
pip install -e ".[eval]"   # adds lm-evaluation-harness
python benchmarks/evaluate.py --tasks ifeval --apply-chat-template --max-tokens 2048
python benchmarks/evaluate.py --tasks mmlu_pro --apply-chat-template
```

### Performance

Measured on an **Apple M3 Max (69 GB)**, size `m` checkpoint, 1024 input / 1024 output tokens,
chunked prefill (256-token chunks), best of 5 runs. `TTFT` = prefill + first token;
`TPS` = steady-state decode throughput; `MLX peak memory` = `mx.get_peak_memory()` (MLX Metal allocator).
References are the matching original `google/gemma-4-*-it` checkpoint served via mlx-vlm: bf16,
and 4-bit quantized (affine, group size 32).

**Gemma 4 E2B**

| Model | TTFT | Decode (TPS) | MLX peak memory |
|---|---|---|---|
| **TheStage (ours)** | **434 ms** | **115.0** | **2.1 GB** |
| Reference bf16 | 531 ms | 57.2 | 10.7 GB |
| Reference 4-bit (gs32) | 595 ms | 83.3 | 4.6 GB |

**Gemma 4 E4B**

| Model | TTFT | Decode (TPS) | MLX peak memory |
|---|---|---|---|
| **TheStage (ours)** | **832 ms** | **73.7** | **3.5 GB** |
| Reference bf16 | 1110 ms | 30.5 | 16.4 GB |
| Reference 4-bit (gs32) | 970 ms | 53.5 | 7.1 GB |

Reproduce:

```bash
python benchmarks/performance.py --model TheStageAI/gemma-4-E2B-it \
    --hf-model google/gemma-4-E2B-it \
    --input-tokens 1024 --output-tokens 1024 --prefill-step-size 256 \
    --compare-ref --compare-ref-4bit --ref-4bit-group-size 32
```

## Acknowledgments

`edge-lm-serve` is a fork of [**edge-lm**](https://github.com/TheStageAI/edge-lm) by [TheStageAI](https://thestage.ai). The compressed Gemma 4 checkpoints, the MLX inference core (the `edge_lm` package), the examples, and the benchmarks above are their work. This project adds the OpenAI-compatible warm-model serving layer (`edge_lm.serve`) on top and tracks upstream via the `upstream` git remote. Model details: [*7× size reduction for Gemma 4 Edge models — Compressing PLE architectures*](https://app.thestage.ai/blog/7x-size-reduction-for-Gemma4-Edge-models?id=14). Huge thanks to the TheStage team.

## License

Released under the [MIT License](LICENSE), © 2026 thestage.ai labs. Serving-layer additions in this fork are likewise MIT.

The compressed model weights are derivatives of Google's Gemma 4 and are additionally subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
