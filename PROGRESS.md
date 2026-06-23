# Progress

Tracking the 7-rung ladder in [plan.md](plan.md). Each rung gets a benchmark run
(`scripts/benchmark_endpoint.py`) and a plot (`scripts/plot_endpoint.py`) archived
under `benchmarks/`. See [README](README.md#benchmarks) for the latest plot.

| # | Rung | Status | Artifact |
|---|------|--------|----------|
| 1 | Naive generation loop — token-by-token, no cache; measure tok/s | ✅ done | — |
| 2 | Add a KV cache; measure the jump | ✅ done | `benchmarks/rung2_kv_cache.{json,png}` |
| 3 | Add batching — several prompts through the loop at once | ✅ done | `benchmarks/rung3_batching.{json,png}` |
| 4 | Quantize one layer by hand (int4 + dequant); measure | ⬜ todo | — |
| 5 | Write one Triton kernel (fused softmax / toy attention) vs PyTorch | ⬜ todo | — |
| 6 | Implement speculative decoding on a toy model (draft + verify) | ⬜ todo | — |
| 7 | Read FlashAttention / vLLM paged attention with primitives built | ⬜ todo | — |

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

### Rung 4 — quantization (next)
- Goal: quantize one layer by hand (int4 weights + dequant), measure the memory saving vs the
  dequant overhead.
