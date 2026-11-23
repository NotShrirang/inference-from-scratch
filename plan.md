# Inference Optimization Plan

How I work: I add each optimization **directly into the FastAPI server** (`app/`), measure it on
the benchmark board, and write down what I learned. Every step changes the server and is
benchmarked. I do not write separate one-off scripts for experiments.

Hardware: one GPU with about 8 GB of memory, so I use small models with normal (softmax) attention,
such as Qwen2.5-0.5B.

Status: ✅ done · 🔎 tried, paused · ⏸ postponed · ⬜ not started. Details and results are in
[PROGRESS.md](PROGRESS.md).

---

## Steps

1. **Simple generation loop.** Generate one token at a time with no cache. `/generate`. ✅
2. **KV cache.** Process the prompt once, then compute only one new token per step, and stream the
   output. `/generate-stream`. ✅
3. **Static batching.** Run N prompts through the model together. `/generate-batch`. ✅
4. **Continuous batching.** A background scheduler and a request queue. Live requests are put into
   one batch: new requests join when they arrive, all active requests are processed together, and
   requests leave when they finish. Each request streams its own output. This gives the biggest
   gain for serving many users. ✅
5. **Paged KV cache.** Store the KV cache in fixed-size blocks with a list of free blocks, so memory
   does not fragment and more requests fit. This is the main idea of PagedAttention, written in
   Python. 🔎
6. **Prefix caching.** Find prompts that start with the same text (for example the same system
   prompt) and reuse their KV cache instead of computing it again. Measure how much faster the
   first token arrives. ⬜
7. **Speculative decoding.** Load a small draft model next to the main model. The draft model guesses
   k tokens, the main model checks them all in one pass, and the correct ones are kept. Measure how
   many guesses are accepted and the change in tokens/s. ⬜
8. **Scheduling and admission control.** Request priorities, a limit on how many requests run at
   once, pausing requests when needed, and **chunked prefill** (processing a long prompt in parts)
   so a long prompt does not slow down everyone else. ✅ (limit + bounded queue + HTTP 429 done)
9. **Serve a quantized model.** Add library quantization (bitsandbytes or AWQ) **and** KV cache
   quantization to `/load`, so the server runs int8/int4 models and the results go on the board.
   ⬜ **next**
10. **Different attention types.** Serve models with GQA, MQA or sliding-window attention through
    `/load`. Measure how a smaller KV cache lets more requests run at once before memory runs out. ⬜
11. **Compilation.** Use `torch.compile` and CUDA graphs when the model loads. Measure the change in
    TTFT and TPOT. I use the compiler only. I do not write GPU kernels by hand. ⏸
12. **Monitoring and latency targets.** Add Prometheus metrics for TTFT, TPOT, throughput, queue
    size and KV cache usage. Plot latency against throughput, and measure **goodput** (the
    throughput of requests that meet a latency target) to choose the best request limit. ⬜
13. **Structured output.** Generate output that follows a JSON schema or grammar. Measure the extra
    cost compared with normal generation. ⬜
14. **OpenAI-compatible API.** Support the OpenAI completions and chat API format, so standard
    clients and load-testing tools work with the server. Optional: send requests to more than one
    server copy. ⬜
15. **Capstone: compare with vLLM.** Run the same model on vLLM, tune its settings (`max-num-seqs`,
    `gpu-memory-utilization`), and benchmark it next to my server. See how close I got and where
    the difference comes from. ✅

---

## Out of scope

This repo does not cover custom GPU kernel work: a real PagedAttention kernel, fused kernels,
FlashAttention, or using more than one GPU. These do not fit a Python/PyTorch server or a single
8 GB GPU. The vLLM comparison showed that most of the speed difference comes from these.
