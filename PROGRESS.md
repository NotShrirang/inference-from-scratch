# Progress

Tracking the 7-rung ladder in [plan.md](plan.md). Each rung gets a benchmark run
(`scripts/benchmark_endpoint.py`) and a plot (`scripts/plot_endpoint.py`) archived
under `benchmarks/`. See [README](README.md#benchmarks) for the latest plot.

| # | Rung | Status | Artifact |
|---|------|--------|----------|
| 1 | Naive generation loop — token-by-token, no cache; measure tok/s | ✅ done | — |
| 2 | Add a KV cache; measure the jump | ✅ done | `benchmarks/rung2_kv_cache.{json,png}` |
| 3 | Add batching — several prompts through the loop at once | ✅ done | `benchmarks/rung3_batching.{json,png}` |
| 4 | Quantize one layer by hand (int4 + dequant); measure | ✅ done | `quntization_script.py`, `infer_quantized.py` |
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
- Memory: int4 ≈ 4× smaller than fp16; quality recoverable only with finer (per-channel/per-group)
  scales. Neither int8 nor int4 is *faster* — dequant (+ int4 unpack) every forward. Speed needs a
  fused kernel → rung 5. Per-group scales (AWQ/GPTQ-grade) are the natural further refinement.

### Rung 5 — Triton kernel (next)
- Goal: write one Triton kernel (fused softmax or toy attention) vs PyTorch — the first real taste
  of the kernel level, and what would let quantized weights actually run faster.
