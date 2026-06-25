# Progress

Tracking the [ladder in plan.md](plan.md). **Build track** = optimizations integrated into the
FastAPI server, each benchmarked on the board. **Read & interview track** = kernel/theory/hardware-
limited topics Claude quizzes me on (no code). See [README](README.md#benchmarks) for plots.

## Build track

| # | Rung | Status | Artifact |
|---|------|--------|----------|
| 1 | Naive generation loop — `/generate` | ✅ done | — |
| 2 | KV cache + real streaming — `/generate-stream` | ✅ done | `benchmarks/rung2_kv_cache.{json,png}` |
| 3 | Static batching — `/generate-batch` | ✅ done | `benchmarks/rung3_batching.{json,png}` |
| 4 | Continuous / in-flight batching (scheduler + queue) | ⬜ **next** | — |
| 5 | Paged / block KV cache (block allocator) | ⬜ todo | — |
| 6 | Prefix caching (shared-prefix KV reuse) | ⬜ todo | — |
| 7 | Speculative decoding (draft + verify) | ⬜ todo | — |
| 8 | Scheduling & admission control (priorities, chunked prefill) | ⬜ todo | — |
| 9 | Serve a quantized model (library quant + KV quant via `/load`) | ⬜ todo | — |
| 10 | Attention-variant model swap (GQA/MQA, sliding window) | ⬜ todo | — |
| 11 | Compilation (`torch.compile` + CUDA graphs) | ⬜ todo | — |
| 12 | Observability & SLO tuning (prometheus + Pareto/goodput) | ⬜ todo | — |
| 13 | Structured / guided decoding (JSON/grammar) | ⬜ todo | — |
| 14 | OpenAI-compatible API | ⬜ todo | — |
| 15 | Capstone — benchmark vs vLLM on the board | ⬜ todo | — |

## Read & interview track (Claude quizzes me — no code)

| Topic | Status |
|-------|--------|
| Quantization theory (per-group, GPTQ/AWQ/SmoothQuant, FP8) | ⬜ — built int4 by hand (`quntization_script.py`), own the theory |
| GPU kernels & Triton (fused attention, dequant-matmul, FlashAttention) | ⬜ — built fused softmax (`softmax_kernel.py`, ~parity) |
| Roofline (arithmetic intensity, BW vs compute bound) | ⬜ todo |
| PagedAttention / FlashAttention internals | ⬜ todo |
| Parallelism (TP/PP/EP, multi-node) — hardware-limited | ⬜ todo |
| Model-level (distillation, pruning, MoE) | ⬜ todo |

## Notes per rung

### Rung 1 — naive loop
- Baseline greedy loop in `ChatModel.generate` / `/generate-stream`, re-encoding the
  full sequence each step.

### Rung 2 — KV cache ✅
- `forward(inputs, past_key_values=None)` threads `use_cache=True` + `past_key_values`
  through prefill → decode; decode feeds only the new token + cache.
- Fixed streaming buffering (`await asyncio.sleep(0)` per token) so TTFT/TPOT are real.
- Result (Qwen2.5-0.5B, max_length=64): TTFT ~23 ms (one prefill pass), TPOT ~18 ms/token
  at concurrency 1. Throughput stays ~60–83 tok/s across concurrency (no batching yet) and
  latency scales ~linearly — the flat throughput line is what rung 3 should bend upward.

### Rung 3 — static batching ✅
- `/generate-batch` takes N prompts in one request; `ChatModel.generate_batch` left-pads,
  tokenizes once, prefills the whole batch, then decodes in lockstep (one new token column
  + shared KV cache per step), decoding ids at the end.
- Result (Qwen2.5-0.5B, max_length=64): total throughput climbs ~12× (98 → 1204 tok/s) from
  batch 1 → 16 while per-request latency stays ~0.8s — textbook memory-bandwidth amortization.
  Batch 32 OOMs on the ~8GB card.
- Benchmark with `scripts/benchmark_batch.py` (sweeps batch size), plot with `scripts/plot_batch.py`.

### Rung 4 — quantization ✅
- `quntization_script.py`: symmetric, **per-channel** weight-only quantization of every nn.Linear
  (`QuantizedModel` swaps them in place). bits=8 stores int8; bits=4 packs two 4-bit values per
  uint8 byte (PyTorch has no int4 dtype). `infer_quantized.py` rebuilds the skeleton + loads the
  saved buffers (plain `from_pretrained` can't — the saved config is stock Qwen2 but the tensors
  are custom `w_quantized`/`scale`).
- Lessons (DeepSeek-R1-Distill-Qwen-1.5B): int8 is coherent even **per-tensor**; int4 **per-tensor
  is word-salad** (one outlier blows up the shared scale → most weights round to 0). Switching to
  **per-channel scales** (one per output row) made int4 coherent (toy rel-err ~1%, vs catastrophic
  per-tensor). Output is coherent-but-lossy at int4 — semantic drift remains.
- Benchmark (DeepSeek-R1-Distill-Qwen-1.5B, 128 tokens, `benchmark_quantized.py`):

  | variant | stored weights | peak VRAM | tok/s |
  |--------:|------:|------:|------:|
  | fp16 | 3554 MB | 3570 MB | 82.3 |
  | int8 | 2247 MB | 2436 MB | 12.8 |
  | int4 | 1591 MB | 1740 MB | 7.4 |

  - **Speed collapsed, not improved**: int8 6.4× slower, int4 11× slower than fp16. Naive dequant
    rebuilds the full fp weight (+ int4 unpack) every forward in eager PyTorch — that dwarfs the
    decode matmul. The "dequant eats the savings" lesson, in full.
  - **Memory dropped only ~2.2× (not 4×)**: embeddings + lm_head stay bf16 (not nn.Linear / skipped)
    and form a large floor for this 150k-vocab model. Peak VRAM ≈ stored size — the dequant transient
    is one layer at a time, freed after each.
  - Quality recoverable only with finer (per-channel/per-group) scales; speed needs a fused kernel
    → rung 5. Per-group scales (AWQ/GPTQ-grade) are the natural further refinement.

### Standalone experiments (done, now off the build path)
- **Quantization** (`quntization_script.py`, `infer_quantized.py`) — per-channel int8/int4 by hand.
  Theory moved to the read/interview track; future quant work is server-integrated (build rung 9).
- **Triton fused softmax** (`softmax_kernel.py`) — correct, ~parity with `torch.softmax` (matching a
  tuned library kernel is the win). Kernel topics now live in the read/interview track.

### Build rung 4 — continuous batching (next)
- Goal: a background scheduler + request queue that merges independent in-flight `/generate-stream`
  requests into a rolling batch (admit on arrival, step together, evict on EOS). Should bend the flat
  concurrency curve from rung 2 upward — re-run `scripts/benchmark_endpoint.py` and compare on the board.
