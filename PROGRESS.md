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
| 11 | Compilation (`torch.compile` + CUDA graphs) | ⏸ deferred (needs static-shape bucketing) | — |
| 12 | Observability & SLO tuning (prometheus + Pareto/goodput) | ⬜ todo | — |
| 13 | Structured / guided decoding (JSON/grammar) | ⬜ todo | — |
| 14 | OpenAI-compatible API | ⬜ todo | — |
| 15 | Capstone — benchmark vs vLLM on the board | ✅ done | `benchmarks/capstone_*.{json,png}` |

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
  tokens out via a per-request `asyncio.Queue`. Endpoint: `/generate-continuous-batch`.
- Cache strategy = **KV cache surgery** (upgraded from the original recompute-on-rebatch):
  - **admit** = prefill *only* the newly-arrived requests (`_prefill` over `new_requests`) and **splice**
    their KV into the active cache (left-pad the seq dim to match, `cat` on the batch dim). In-flight
    requests keep their KV — no re-prefill of the whole batch on every admit.
  - **evict** = slice finished rows out of the cache/mask (`index_select`), then `_trim_left_pad` drops
    leading columns that are padding for every remaining row (else the shared seq dim only grows).
  - **positions** = pass mask-derived `position_ids` on every forward. RoPE is baked into the stored
    keys at forward time, so a spliced request needs its *own* position (from its mask), not the shared
    cache length — without this, spliced requests decode at the wrong RoPE offset (subtly wrong output,
    verified `Paris is` vs correct `Paris.`).
  - transformers 5.x note: `DynamicCache` exposes per-layer tensors at `.layers[l].keys/.values`
    (not the old `.key_cache/.value_cache` lists).
- **The OOM that dogged every run was `no_grad`, not the cache logic.** `model.forward` ran without
  `torch.no_grad()`, so each step built an autograd graph; since the KV cache is a graph output fed back
  in, the whole graph + activations stayed pinned and grew every decode token → VRAM climbed to 7.6 GB
  on a 0.5B model → OOM. One decorator (`@torch.no_grad()` on `ChatModel.forward`) fixed it and also
  shaved graph-building overhead. `eval()` mode does **not** disable grad tracking — a separate concern.
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
  - _(Numbers above are the original recompute-on-rebatch build — `step4_continuous_batching.{json,png}`.
    The post-fix engine (cache surgery + `no_grad`) peaks **~788 tok/s @ conc 64**, 0 errors through 64,
    sheds 429 beyond — see the capstone table and `capstone_custom.json`.)_
  - Bug trail that got it working: persist the cache across steps; grow the attn mask once per step
    (guarded, skip after eviction); set `padding_side` on the tokenizer; finish on *generated* count;
    re-prefill from full `req.tokens` (not the prompt) so rebatch keeps context; sampling (not argmax)
    to avoid the base model's greedy echo.

### Step 5 — paged / block KV cache 🔎 explored, not shipped (→ read track)
- The two cache-surgery pieces on HF's `DynamicCache` are now **shipped in the step-4 engine**: **evict**
  = slice finished rows out (`index_select` + `_trim_left_pad`); **admit** = prefill only the new request
  and **splice** its KV in (left-pad seq dim, `cat` on batch dim). What it took to make them correct:
  mask-derived `position_ids` (RoPE offset), `_trim_left_pad` (bound the growing seq dim), and per-layer
  `.layers[l].keys/.values` access (transformers 5.x). Fragile but working — this is *cache surgery*, not
  *paging*: the cache is still one contiguous rectangular tensor, so it still pads and reallocates.
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
- Hardened `_run`: `_step` is wrapped in `try/except` → on failure it logs, `_reset_batch()` (fails the
  in-flight requests with a sentinel, drops the cache, `empty_cache()`), and continues — one bad step no
  longer silently bricks the engine loop. (Still optional: `Retry-After` on the 429, `uvicorn
  --limit-concurrency` front door.)
- Debugging-tail lessons (all env/tooling, not the engine): (1) `/load` reloaded weights to VRAM every
  call → session VRAM creep → OOM; fixed by checking already-loaded before constructing. (2) The benchmark
  used `timeout=None`, so a starved stream hung the whole level; fixed with a per-request timeout + a hard
  level deadline that cancels stragglers. (3) A long-lived server fragments VRAM across back-to-back heavy
  sweeps → 30× slowdown until restart — restart between stress runs (or `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`). This fragility is exactly what fixed-size paged pools avoid — memory
  determinism, again.

## Profiling & micro-optimizations

Tooling: `scripts/profile_request.py` (standalone) + `scripts/profile_service.py` (drives the live
engine) + `/engine-profile` endpoint. Per-op timing with `torch.cuda.synchronize()` (so async kernels
attribute correctly), averaged over N requests.

- **Where the time goes** (Qwen2.5-0.5B, 100 req, conc 8): **decode_fwd ~69%** (runs ~64× per request),
  prefill ~17–23% (once per request + recompute-on-rebatch churn), sample ~8%, tokenize ~0%. Decode is
  the irreducible cost — concrete proof that TPOT is *the* metric.
- **Batched sampling (kept):** vectorized top-p over `[batch, vocab]` instead of a per-request Python
  loop. Clean same-GPU A/B: **~1.9× faster** sampling (2100 → 1124 ms over 100 req), 14% → 8% of the
  pipeline, ~7% off whole-run. Win = one batched sort + one `.tolist()` instead of B sorts + B `.item()`
  syncs per step. Residual cost is the full-152k-vocab sort (top-k pre-filter would shrink it further).
- **torch.compile explored → deferred (step 11):** `mode="reduce-overhead"` (CUDA graphs) on the
  dynamic-shape decode loop recompiled ~every step → **15× slower** (decode 12 → 178 ms, run 13 s → 6 min).
  Compile needs static-shape bucketing (fixed batch + seq buckets) to pay off; deferred to the read track
  with paging. Note: the recurring OOM turned out to be the missing `torch.no_grad()` in `forward`
  (autograd graph pinned by the growing KV cache — see Step 4 note), *not* the separate `/load` bug
  (constructs `ChatModel` before the already-loaded check → leaks VRAM across a session); both are real,
  but `no_grad` was the one filling the card.

## Step 15 — capstone: custom engine vs vLLM ✅

Same model (Qwen2.5-0.5B), same prompts, same metrics; each server benchmarked **alone** on the 8 GB
GPU (they can't co-reside — vLLM grabs ~90% VRAM). `benchmarks/capstone_{custom,vllm}.json`,
`capstone_vllm_vs_custom.png` (dedicated board `benchmarks/capstone.json`).

| metric | Custom | vLLM | gap |
|--------|------:|------:|----:|
| tok/s @ conc 1 | 90 | 270 | 3.0× |
| tok/s peak | 788 (@64) | 9370 (@256) | ~12× |
| TTFT p50 @ conc 1 | 114 ms | 9 ms | 13× |
| TPOT p95 @ conc 1 | 8.8 ms | 3.7 ms | 2.4× |
| errors @ conc 256 | 902 (sheds) | 0 | — |

_(Custom numbers are the post-fix engine: cache surgery + `no_grad` — see Step 4/5 notes.)_

Gap attribution (the payoff — every piece maps to a deferred technique):
- **~12× throughput + never shedding → PagedAttention** (step 5): vLLM packs hundreds of sequences into
  8 GB (fixed blocks, no padding/fragmentation), batch grows to 256+; ours caps at 16 and sheds.
- **TPOT ~2.4× → fused kernels + CUDA graphs + FlashAttention** (step 11 / kernel track): vLLM's decode is
  one graph-captured fused kernel; ours is eager per-op PyTorch.
- **TTFT 13× → fast prefill kernel + leaner serving path.** Originally suspected the `stream_generator`'s
  `await asyncio.sleep(0.1)` poll, but removing it barely moved TTFT (119 → ~114 ms) — the gap is
  fundamental: eager PyTorch prefill + a Python/ASGI serving path vs vLLM's fused prefill + C++ path.

Verdict: paging owns the capacity/throughput axis; fused kernels own the per-token-latency axis — both
are the read/interview-track items, now understood from having built a server without them.

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