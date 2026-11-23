# Progress

My notes for each step in [plan.md](plan.md). Each step is built into the FastAPI server and
benchmarked on the board. The plots are in the [README](README.md#results).

## Status

| # | Step | Status | Files |
|---|------|--------|----------|
| 1 | Simple generation loop: `/generate` | ✅ done | — |
| 2 | KV cache + streaming: `/generate-stream` | ✅ done | `benchmarks/step2_kv_cache.{json,png}` |
| 3 | Static batching: `/generate-batch` | ✅ done | `benchmarks/step3_batching.{json,png}` |
| 4 | Continuous batching (scheduler + queue) | ✅ done | `app/batching/engine.py`, `benchmarks/step4_continuous_batching.{json,png}` |
| 5 | Paged KV cache (block allocator) | 🔎 tried, paused | see notes |
| 6 | Prefix caching (reuse KV cache for shared prompt starts) | ⬜ not started (now possible, because KV cache editing works) | — |
| 7 | Speculative decoding (draft + check) | ⬜ not started (now possible, because KV cache editing works) | — |
| 8 | Admission control (batch limit + bounded queue → 429) | ✅ done | `benchmarks/step8_admission_control.{json,png}` |
| 9 | Serve a quantized model (library quantization + KV quantization in `/load`) | ⬜ **next** | — |
| 10 | Different attention types (GQA/MQA, sliding window) | ⬜ not started | — |
| 11 | Compilation (`torch.compile` + CUDA graphs) | ⏸ postponed (needs fixed input shapes) | — |
| 12 | Monitoring and latency targets (Prometheus + goodput) | ⬜ not started | — |
| 13 | Structured output (JSON/grammar) | ⬜ not started | — |
| 14 | OpenAI-compatible API | ⬜ not started | — |
| 15 | Capstone: compare with vLLM on the board | ✅ done | `benchmarks/capstone_*.{json,png}` |

## Notes for each step

### Step 1: simple loop ✅
- A basic loop in `ChatModel.generate` / `/generate-stream` that processes the full text again for
  every new token. It is slow on purpose. Every other step is compared with it.

### Step 2: KV cache ✅
- `forward(inputs, past_key_values=None)` passes `use_cache=True` and `past_key_values` from the
  prompt step to each generation step. Each generation step sends only the new token and the cache.
- My first TTFT and TPOT numbers were wrong, because the stream was being buffered. Adding
  `await asyncio.sleep(0)` after each token fixed it.
- Results (Qwen2.5-0.5B, max_length=64): TTFT about 23 ms (the prompt is processed once), TPOT about
  18 ms per token with one request. Throughput stays at about 60–83 tokens/s for all load levels,
  because there is no batching yet. Latency grows in a straight line with load.
- What I learned: the cache makes one request faster, but it does not help with many requests at
  the same time. Batching is needed for that.

### Step 3: static batching ✅
- `/generate-batch` takes N prompts in one request. `ChatModel.generate_batch` pads the prompts on
  the left, tokenizes them together, processes them all at once, and then generates one token for
  every prompt at each step, using a shared KV cache. At the end it turns the token ids back into text.
- Results (Qwen2.5-0.5B, max_length=64): total throughput goes up about 12× (98 → 1204 tokens/s)
  from batch size 1 to 16. The time for each request stays about 0.8 s. Batch size 32 runs out of
  memory on my 8 GB GPU.
- What I learned: the GPU reads the model weights once per step, no matter how many sequences are
  in the batch. So more sequences cost very little extra, until memory runs out.
- I benchmark with `scripts/benchmark_batch.py` (tests different batch sizes) and plot with
  `scripts/plot_batch.py`.

### Step 4: continuous batching ✅
- `app/batching/engine.py`: one background loop controls the model. Requests are added to a queue,
  and each request gets its tokens back through its own `asyncio.Queue`. Endpoint:
  `/generate-continuous-batch`.
- KV cache handling. My first version recomputed the whole batch whenever it changed. Now the engine
  edits the KV cache directly:
  - **New requests:** process only the new prompts (`_prefill` over `new_requests`) and add their
    KV cache to the existing one. I pad the sequence dimension on the left so the sizes match, and
    join them with `cat` on the batch dimension. Running requests keep their cache.
  - **Finished requests:** remove their rows from the cache and mask with `index_select`. Then
    `_trim_left_pad` removes columns at the start that are only padding for all remaining rows.
    Without this step, the cache only gets longer.
  - **Positions:** calculate `position_ids` from the attention mask on every forward pass. RoPE
    (the position encoding) is applied to the keys when they are stored, so each added request needs
    its own position, not the length of the shared cache. Without this, added requests used the
    wrong positions. The output was only a little wrong: I got `Paris is` instead of `Paris.`. This
    kind of bug is dangerous because nothing crashes.
  - Note for transformers 5.x: `DynamicCache` stores each layer's tensors at
    `.layers[l].keys/.values`, not in the old `.key_cache/.value_cache` lists.
- **The out-of-memory error in every run was caused by a missing `no_grad`, not by my cache code.**
  `model.forward` ran without `torch.no_grad()`, so PyTorch saved data for gradients at every step.
  The KV cache is part of that saved data and goes back into the model, so all of it stayed in
  memory and grew with every token. GPU memory went up to 7.6 GB with a 0.5B model and then
  crashed. Adding `@torch.no_grad()` to `ChatModel.forward` fixed it and also made each step a bit
  faster. Lesson: `eval()` mode does **not** turn off gradient tracking. They are two separate things.
- Results (Qwen2.5-0.5B, sampling) compared with step 2, at 16 concurrent requests:

  | metric at 16 requests | step 2 (one at a time) | step 4 (continuous batching, batch_size=16) |
  |--------|------:|------:|
  | throughput | 86 tokens/s (flat) | 389 tokens/s (about 8.3× higher) |
  | TPOT p95 | 0.179 s | 0.019 s (same at all load levels) |
  | total latency p99 | 12.1 s | 2.3 s |

  - Full test (random prompts, 1 to 256 concurrent requests): throughput **goes up until 16
    requests (the batch size), then stays flat at about 383 tokens/s**. **TPOT stays about 0.02 s at
    all load levels**, because each token costs the same once a request is in the batch. TTFT and
    total latency stay low until 16 requests, then grow in a straight line as requests wait for a
    free place (TTFT 0.08 s → 25 s at 256 requests).
  - What I learned: after the batch is full, more load only adds waiting time, not throughput. So
    the best load for good latency is about the batch size.
  - With one request, this engine is a little *slower* than step 2 (47 vs 86 tokens/s). The engine
    has some extra work, and there is no batch to share it with.
  - _(These numbers are from my first version, which recomputed the batch:
    `step4_continuous_batching.{json,png}`. The fixed engine (KV cache editing + `no_grad`) reaches
    **about 788 tokens/s at 64 requests**, has 0 errors up to 64, and returns 429 above that. See
    the capstone table and `capstone_custom.json`.)_
  - Bugs I fixed to make it work: keep the cache between steps; extend the attention mask once per
    step (and not after removing requests); set `padding_side` on the tokenizer; stop based on the
    number of *generated* tokens; recompute from the full `req.tokens`, not only the prompt, so no
    text is lost when the batch changes; and use sampling instead of always picking the top token,
    because the base model repeats itself with greedy decoding.

### Step 5: paged KV cache 🔎 tried, not finished
- The two KV cache editing parts on HuggingFace's `DynamicCache` are now **used in the step 4
  engine**: removing finished rows (`index_select` + `_trim_left_pad`) and adding new requests
  (process only the new prompt, pad on the left, `cat` on the batch dimension). To make them correct
  I needed `position_ids` from the mask, `_trim_left_pad` to stop the cache from growing forever,
  and the per-layer `.layers[l].keys/.values` access in transformers 5.x. It is fragile, but it
  works. However, this is not *paging*. The cache is still one large tensor, so it still has padding
  and still needs new memory when it grows.
- I also tried to **choose `batch_size` automatically** from the free GPU memory. My calculation
  (weights + KV cache + activations) was far too high: it said 8192. Measuring one real prompt step
  was also too high: it said 230, but the real limit was about 32.
- **Why I paused it:** a Python-only version on top of HuggingFace cannot do better. Real paging needs
  a custom attention kernel that can read the KV cache from blocks in different places in memory,
  and custom kernels are out of scope for this repo. The main benefit of paging is **predictable
  memory use**, not speed. With fixed-size blocks there is no fragmentation, so peak memory is
  exactly `num_blocks × block_bytes`. With my approach I could not calculate peak memory reliably,
  and that is the reason PagedAttention exists. I learned this by trying and failing, and it was
  still useful.

### Step 8: admission control ✅
- A `batch_size` limit on the active batch and a limited `waiting_requests` queue
  (`max_waiting = 4 × batch_size`). `submit` raises an error when the queue is full, and the
  endpoint returns **HTTP 429**. At first the 429 came back as a 500. The reason: `HTTPException`
  is an `Exception`, so the general error handler must raise it again, not wrap it.
- Results compared with step 4 (both with batch_size=16, tested up to 256 requests): my earlier
  version with automatic batch size crashed at 32 requests. With admission control it **keeps
  working up to 256**. Throughput stays at about 450–570 tokens/s, TPOT stays at about 0.03 s, and
  extra requests get a 429 (2430 at 128 requests, 1367 at 256).
- My opinion: the benefit here is not more throughput. It is a **limited queue that rejects extra
  requests clearly**, instead of a queue that grows without limit until the server crashes. I
  prefer to reject some requests than to fail all of them.
- I made `_run` more reliable: `_step` is inside a `try/except`. If it fails, the engine logs the
  error, calls `_reset_batch()` (marks the running requests as failed, clears the cache, and runs
  `empty_cache()`), and continues. One failed step no longer stops the engine loop without any
  message. (Still optional: a `Retry-After` header on the 429, and `uvicorn --limit-concurrency`
  to limit connections before they reach the app.)
- Problems I found while debugging (all in setup and tools, not in the engine):
  1. `/load` loaded the weights to the GPU again on every call, so memory use grew over time until
     it crashed. I fixed it by checking if the model is already loaded first.
  2. The benchmark used `timeout=None`, so one stuck request could stop a whole test level. I added
     a timeout for each request and a time limit for each level that cancels unfinished requests.
  3. A server that runs for a long time fragments GPU memory after many heavy tests, and I saw a
     30× slowdown until I restarted it. Now I restart between heavy tests (or set
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). A paged KV cache with fixed-size blocks
     avoids this problem. This again shows why predictable memory use matters.

## Profiling and small optimizations

Tools: `scripts/profile_service.py` (sends requests to the running engine) and the
`/engine-profile` endpoint. I time each operation with `torch.cuda.synchronize()`, so the GPU time
is counted in the right place, and I average over N requests.

- **Where the time goes** (Qwen2.5-0.5B, 100 requests, 8 at a time): **generating tokens
  (`decode_fwd`) about 69%** (it runs about 64 times per request), processing prompts about 17–23%
  (once per request, plus recomputing when the batch changes), sampling about 8%, tokenizing about
  0%. Generating tokens is the cost I cannot avoid. This showed me that TPOT is the most important
  number to watch.
- **Batched sampling (kept):** top-p sampling for the whole `[batch, vocab]` tensor at once, instead
  of a Python loop over requests. In a fair test on the same GPU, sampling became **about 1.9×
  faster** (2100 → 1124 ms over 100 requests). Its share went from 14% to 8% of the total time, and
  the whole run became about 7% faster. The reason: one sort and one `.tolist()` for the batch,
  instead of B sorts and B `.item()` calls per step. Most of the remaining cost is sorting the full
  152k-token vocabulary. Filtering to the top-k tokens first would make this faster.
- **torch.compile tried, postponed to step 11:** `mode="reduce-overhead"` (CUDA graphs) recompiled
  almost every step, because the input shapes keep changing. It was **15× slower** (generation step
  12 → 178 ms, full run 13 s → 6 min). Compilation only helps with fixed input shapes (fixed batch
  sizes and sequence lengths). Also: the repeated out-of-memory error was caused by the missing
  `torch.no_grad()` in `forward` (see the step 4 notes), not by the separate `/load` bug that used
  more GPU memory over time. Both bugs were real, but `no_grad` was the one that filled the GPU.

## Step 15: capstone, my engine vs vLLM ✅

Same model (Qwen2.5-0.5B), same prompts and same measurements. I tested each server **alone** on
the 8 GB GPU, because vLLM takes about 90% of GPU memory and they cannot run together. Files:
`benchmarks/capstone_{custom,vllm}.json` and `capstone_vllm_vs_custom.png` (it has its own board,
`benchmarks/capstone.json`).

| metric | Mine | vLLM | difference |
|--------|------:|------:|----:|
| tokens/s with 1 request | 90 | 270 | 3.0× |
| peak tokens/s | 788 (at 64) | 9370 (at 256) | about 12× |
| TTFT p50 with 1 request | 114 ms | 9 ms | 13× |
| TPOT p95 with 1 request | 8.8 ms | 3.7 ms | 2.4× |
| errors at 256 requests | 902 (rejected) | 0 | — |

_(My numbers are from the fixed engine with KV cache editing and `no_grad`. See the notes for
steps 4 and 5.)_

Where the difference comes from (for me this was the most useful part, because each reason is a
feature my engine does not have):
- **About 12× more throughput and no rejected requests: PagedAttention** (step 5). vLLM fits hundreds
  of sequences in 8 GB, because it uses fixed-size blocks with no padding and no fragmentation. Its
  batch grows to 256 or more. Mine is limited to 16 and rejects extra requests.
- **About 2.4× lower TPOT: fused kernels, CUDA graphs and FlashAttention** (step 11 covers the
  compilation part). vLLM runs each generation step as a few combined GPU operations. Mine runs
  normal PyTorch, one operation at a time.
- **13× lower TTFT: faster prompt processing and a faster server code path.** At first I thought the
  `await asyncio.sleep(0.1)` wait in `stream_generator` was the cause. But removing it changed TTFT
  very little (119 → about 114 ms). The real reason is that my server processes prompts with normal
  PyTorch and handles requests in Python, while vLLM uses fused GPU code and a C++ path.

**My conclusion:** PagedAttention decides how many requests fit and how much throughput you get.
Fused kernels decide how fast each token is. Both need custom GPU kernels, so they are out of scope
for this repo. But I now understand why vLLM needs them, because I built a server without them and
saw each problem myself.
