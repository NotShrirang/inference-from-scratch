# Progress

Tracking the [ladder in plan.md](plan.md). **Build track** = optimizations integrated into the
FastAPI server, each benchmarked on the board. **Read & interview track** = kernel/theory/hardware-
limited topics Claude quizzes me on (no code). See [README](README.md#benchmarks) for plots.

## Build track

| # | Step | Status | Artifact |
|---|------|--------|----------|
| 1 | Naive generation loop — `/generate` | ✅ done | — |
| 2 | KV cache + real streaming — `/generate-stream` | ✅ done | `benchmarks/step2_kv_cache.{json,png}` |
| 3 | Static batching — `/generate-batch` | ✅ done | `benchmarks/step3_batching.{json,png}` |
| 4 | Continuous / in-flight batching (scheduler + queue) | ✅ done | `app/batching/engine.py`, `benchmarks/step4_continuous_batching.{json,png}` |
| 5 | Paged / block KV cache (block allocator) | 🔎 explored → read track | see note |
| 6 | Prefix caching (shared-prefix KV reuse) | ⏸ deferred (needs cache surgery) | — |
| 7 | Speculative decoding (draft + verify) | ⏸ deferred (needs cache surgery) | — |
| 8 | Admission control (batch cap + bounded queue → 429) | ✅ done | `benchmarks/step8_admission_control.{json,png}` |
| 9 | Serve a quantized model (library quant + KV quant via `/load`) | ⬜ **next** | — |
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

## Notes per step

### Step 1 — naive loop ✅
- Baseline greedy loop in `ChatModel.generate` / `/generate-stream`, re-encoding the
  full sequence each step.

### Step 2 — KV cache ✅
- `forward(inputs, past_key_values=None)` threads `use_cache=True` + `past_key_values`
  through prefill → decode; decode feeds only the new token + cache.
- Fixed streaming buffering (`await asyncio.sleep(0)` per token) so TTFT/TPOT are real.
- Result (Qwen2.5-0.5B, max_length=64): TTFT ~23 ms (one prefill pass), TPOT ~18 ms/token
  at concurrency 1. Throughput stays ~60–83 tok/s across concurrency (no batching yet) and
  latency scales ~linearly — the flat throughput line is what batching should bend upward.

### Step 3 — static batching ✅
- `/generate-batch` takes N prompts in one request; `ChatModel.generate_batch` left-pads,
  tokenizes once, prefills the whole batch, then decodes in lockstep (one new token column
  + shared KV cache per step), decoding ids at the end.
- Result (Qwen2.5-0.5B, max_length=64): total throughput climbs ~12× (98 → 1204 tok/s) from
  batch 1 → 16 while per-request latency stays ~0.8s — textbook memory-bandwidth amortization.
  Batch 32 OOMs on the ~8GB card.
- Benchmark with `scripts/benchmark_batch.py` (sweeps batch size), plot with `scripts/plot_batch.py`.

### Step 4 — continuous / in-flight batching ✅
- `app/batching/engine.py`: one background loop owns the model; requests submit to a queue and stream
  tokens out via a per-request `asyncio.Queue`. Cache strategy = **recompute-on-rebatch**: pure decode
  steps reuse a merged KV cache; membership changes (admit/evict) set `dirty` → re-prefill over the
  active set's *full* sequences (no DynamicCache surgery). Endpoint: `/generate-continuous-batch`.
- Result (Qwen2.5-0.5B, sampling) vs step-2 baseline, concurrency sweep @ conc 16:

  | metric @ conc 16 | step 2 (per-request) | step 4 (continuous batch, batch_size=16) |
  |--------|------:|------:|
  | throughput | 86 tok/s (flat) | 389 tok/s (climbs ~8.3×) |
  | TPOT p95 | 0.179 s | 0.019 s (flat at every conc) |
  | e2e p99 | 12.1 s | 2.3 s |

  - Canonical serving curve (random varied prompts, conc swept 1→256): throughput **climbs to the knee
    at conc = batch_size (16) then saturates flat at ~383 tok/s**; **TPOT stays ~0.02s across the entire
    range** (constant per-token cost once batched); TTFT/e2e are low until the knee, then grow ~linearly
    as requests queue for a slot (TTFT 0.08s → 25s at conc 256). Lesson: past the knee you buy only
    latency, no throughput — SLO operating point ≈ the knee.
  - At concurrency 1 CB is slightly *slower* (47 vs 86 tok/s) — engine overhead with no batch to amortize.
  - Bug trail that got it working: persist the cache across steps; grow the attn mask once per step
    (guarded, skip after eviction); set `padding_side` on the tokenizer; finish on *generated* count;
    re-prefill from full `req.tokens` (not the prompt) so rebatch keeps context; sampling (not argmax)
    to avoid the base model's greedy echo.

### Step 5 — paged / block KV cache 🔎 explored, not shipped (→ read track)
- Attempted the two pieces on HF's `DynamicCache`: **evict** = slice finished rows out; **admit** =
  prefill only the new request and **splice** its KV in (left-pad seq dim, cat on batch dim). Both are
  doable but require reaching into `DynamicCache.key_cache`/`.value_cache` internals — fragile.
- Also tried **auto-sizing `batch_size`** from VRAM: analytical (weights + KV + activation) over-predicted
  wildly (8192!), a profiling probe of one prefill still over-predicted (230 vs real OOM at ~32).
- **Why it's parked:** the pure-Python-on-HF version can't win. True paging needs a scatter-read
  attention **kernel** (read track) so KV lives in fixed non-contiguous blocks. The payoff of paging is
  **memory determinism**, not speed: fixed blocks kill fragmentation → peak memory becomes exactly
  `num_blocks × block_bytes`; and no recompute-on-rebatch means the only transient is one decode step.
  On a fragmenting allocator with variable-length KV + recompute transients, peak memory is *not*
  perfectly calculable — which is the whole reason PagedAttention exists. Lesson learned; build deferred
  to the kernel/interview track.

### Step 8 — admission control ✅
- `batch_size` cap on the active batch + bounded `waiting_requests` (`max_waiting = 4 × batch_size`);
  `submit` raises when the queue is full → endpoint returns **HTTP 429** (fixed the swallowed-into-500
  bug: `HTTPException` is an `Exception`, so the generic handler must re-raise it, not wrap it).
- Result vs step 4 (both batch_size=16, sweep to conc 256): where the earlier auto-sized run OOM'd at 32,
  admission control **degrades gracefully to 256** — throughput holds ~450–570 tok/s, TPOT flat ~0.03s,
  and overflow is shed as clean 429s (2430 @ 128, 1367 @ 256). The win isn't higher throughput; it's a
  **bounded queue + explicit load-shedding** instead of an unbounded queue / OOM.
- Steps 6 (prefix caching) & 7 (speculative decoding) deferred — both need the same HF `DynamicCache`
  surgery parked in step 5 (splice a shared prefix / roll back rejected draft tokens).
- Optional hardening not done: `Retry-After` on the 429, `uvicorn --limit-concurrency` front door,
  `try/except` around `_step` so one bad step can't kill the engine loop.

## Standalone experiments (done, off the build path)

### Quantization — `quntization_script.py`, `infer_quantized.py`
- Symmetric, **per-channel** weight-only quantization of every nn.Linear (`QuantizedModel` swaps them
  in place). bits=8 stores int8; bits=4 packs two 4-bit values per uint8 byte (PyTorch has no int4
  dtype). `infer_quantized.py` rebuilds the skeleton + loads the saved buffers (plain `from_pretrained`
  can't — the saved config is stock Qwen2 but the tensors are custom `w_quantized`/`scale`).
- Lessons (DeepSeek-R1-Distill-Qwen-1.5B): int8 is coherent even **per-tensor**; int4 **per-tensor is
  word-salad** (one outlier blows up the shared scale → most weights round to 0). **Per-channel scales**
  (one per output row) made int4 coherent (toy rel-err ~1%). int4 output is coherent-but-lossy.
- Benchmark (128 tokens, `benchmark_quantized.py`):

  | variant | stored weights | peak VRAM | tok/s |
  |--------:|------:|------:|------:|
  | fp16 | 3554 MB | 3570 MB | 82.3 |
  | int8 | 2247 MB | 2436 MB | 12.8 |
  | int4 | 1591 MB | 1740 MB | 7.4 |

  - **Speed collapsed, not improved**: int8 6.4× slower, int4 11× slower than fp16. Naive dequant
    rebuilds the full fp weight (+ int4 unpack) every forward in eager PyTorch — dwarfs the matmul.
  - **Memory dropped only ~2.2× (not 4×)**: bf16 embeddings + lm_head are a large floor (150k vocab).
  - Quality needs finer scales; speed needs a fused kernel. → quantization theory is now a read/interview
    topic; serving a quantized model is build step 9.

### Triton fused softmax — `softmax_kernel.py`
- Correct (`allclose` across sizes), ~parity with `torch.softmax` (matching a tuned library kernel is
  the win; softmax is already fused, so no speedup expected). Kernel topics now live in the read track.