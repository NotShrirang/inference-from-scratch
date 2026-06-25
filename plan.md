# Inference Optimization Ladder

Philosophy: build each optimization **into the FastAPI server** (`app/`), measure it on
the benchmark board, then read the paper that formalizes it. Own the primitive before the name.

Two tracks:
- **Build track** — every rung changes the server and is benchmarked. No standalone scripts.
- **Read & interview track** — kernel-level, hardware-limited, or theory-heavy topics that don't
  belong in the server (or can't run on one 8 GB GPU). No code — Claude interviews me to check I
  actually understand them.

Hardware: single ~8 GB GPU; small softmax-attention models (Qwen2.5-0.5B, etc.).

---

## Build track — integrated into the FastAPI server

1. **Naive generation loop** — token-by-token, no cache. `/generate`. ✅
2. **KV cache** — prefill + single-token decode + real streaming. `/generate-stream`. ✅
3. **Static batching** — N prompts through one forward pass. `/generate-batch`. ✅
4. **Continuous / in-flight batching** — a background scheduler + request queue that merges
   independent in-flight requests into a rolling batch: admit on arrival, step all active
   sequences together, evict on EOS, stream each out. The single biggest serving win — turns the
   flat concurrency curve from rung 2 into a climbing one.
5. **Paged / block KV cache** — a block allocator (fixed-size blocks + free list) in the model
   manager so the cache stops fragmenting and more sequences fit. PagedAttention's *idea*, in Python.
6. **Prefix caching** — detect shared prompt prefixes (e.g. a common system prompt) across requests
   and reuse their KV instead of recomputing. Measure the TTFT collapse on shared-prefix workloads.
7. **Speculative decoding** — load a small draft model beside the target; draft k tokens, verify in
   one target forward pass, accept the longest correct run. Measure acceptance rate and tok/s gain.
8. **Scheduling & admission control** — request priorities, a max-concurrent-sequences cap,
   preemption, and **chunked prefill** so a long prompt doesn't stall decode or blow up everyone's TTFT.
9. **Serve a quantized model** — wire library quantization (bitsandbytes / AWQ) **and** KV-cache
   quantization into `/load`, so the endpoint actually serves int8/int4 and it lands on the board.
   (Quantization becomes a serving feature, not a one-off script.)
10. **Attention-variant model swap** — serve GQA / MQA / sliding-window models via `/load`; measure
    how a smaller KV cache raises the max number of concurrent sequences before OOM.
11. **Compilation** — apply `torch.compile` + CUDA graphs at load time; measure the TTFT/TPOT change
    (especially the launch-overhead win at decode). Use the compiler — don't write kernels.
12. **Observability & SLO tuning** — wire `prometheus.yml` + middleware to export
    TTFT / TPOT / throughput / queue-depth / KV-utilization; build the latency–throughput Pareto and
    a **goodput** (SLO-aware) view to pick the right max-concurrency.
13. **Structured / guided decoding** — JSON / grammar-constrained generation on the endpoint (logit
    processors / outlines-style); measure the overhead vs free generation.
14. **OpenAI-compatible API** — make the server speak the OpenAI completions/chat protocol so standard
    clients and load tools work against it; optional multi-replica request routing.
15. **Capstone — measure against vLLM** — deploy the same model on vLLM (OpenAI-compatible), tune it
    (`max-num-seqs`, `gpu-memory-utilization`), and benchmark it on the board next to the hand-built
    server. See how close you got and exactly where the remaining gap lives.

---

## Read & interview track (Claude quizzes me — no code, no server changes)

These are real and important, but they're either kernel-level, theory, or need hardware I don't have.
Claude tests my understanding instead of me building them.

- **Quantization theory** — symmetric/asymmetric, per-tensor/per-channel/per-group, GPTQ / AWQ /
  SmoothQuant, FP8. *(Built int4-by-hand as a standalone experiment ✅ — now own the theory.)*
- **GPU kernels & Triton** — fused softmax ✅ (standalone), fused attention, fused dequant-matmul,
  the FlashAttention tiling/online-softmax trick.
- **Roofline model** — arithmetic intensity, memory-bandwidth vs compute bound, why decode is
  bandwidth-bound and prefill is compute-bound.
- **PagedAttention & FlashAttention internals** — the kernel mechanics under rungs 5 and the attention path.
- **Parallelism** *(hardware-limited: 1 GPU)* — tensor / pipeline / expert parallelism, multi-node,
  prefill/decode disaggregation.
- **Model-level** — distillation (running a distilled model already), pruning/sparsity, MoE serving,
  choosing inference-friendly architectures.

---

## Papers (read once the matching rung is built)

- Orca — continuous batching → rung 4
- vLLM / PagedAttention → rungs 5–6
- Speculative decoding; Medusa / EAGLE → rung 7
- AWQ / GPTQ / SmoothQuant → rung 9
- GQA → rung 10
- FlashAttention → kernel (interview) track

---

## Parked: standalone experiments already done

Kept for reference, but off the main (server) path:
- `quntization_script.py` + `infer_quantized.py` — by-hand int8/int4 weight quantization (rung-4 era).
- `softmax_kernel.py` — Triton fused softmax, ~parity with `torch.softmax`.
