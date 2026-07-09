#!/usr/bin/env python3
"""Benchmark a vLLM (OpenAI-compatible) server with the SAME metrics + JSON schema
as benchmark_endpoint.py, so the result drops straight onto the board for a
side-by-side against your hand-built engine.

vLLM speaks the OpenAI API: POST /v1/completions with a JSON body and streams
`data: {json}\\n\\n` chunks terminated by `data: [DONE]` (unlike your server's
`?params` + `data: <token>`). This client parses that format.

Run vLLM in a SEPARATE venv/port so it doesn't clash with your project:
    uv venv .venv-vllm && source .venv-vllm/bin/activate && uv pip install vllm
    vllm serve Qwen/Qwen2.5-0.5B --port 8001 --max-model-len 2048

Then:
    python scripts/benchmark_vllm.py --model Qwen/Qwen2.5-0.5B \
        --base http://localhost:8001 --out benchmarks/capstone_vllm.json
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

PROMPTS = [
    "What is a KV cache?",
    "Explain how continuous batching improves LLM serving throughput.",
    "Write a haiku about GPUs.",
    "List three techniques to reduce LLM inference latency.",
    "Why is autoregressive decoding memory-bandwidth bound rather than compute bound?",
    "Summarize the difference between the prefill and decode phases of inference.",
    "Describe weight quantization in one short paragraph.",
    "What problem does PagedAttention solve, and how?",
    "Give a short analogy that explains speculative decoding.",
    "How does batching amortize the cost of loading model weights from VRAM?",
]


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


async def one_request(client, base, model, max_tokens):
    payload = {
        "model": model, "prompt": random.choice(PROMPTS), "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.9, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, last, itls, out, ok, err = None, t0, [], 0, False, None
    try:
        async with client.stream("POST", f"{base}/v1/completions", json=payload) as r:
            ok = r.status_code == 200
            if not ok:
                await r.aread()
                err = f"HTTP {r.status_code}"
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    ch = chunk.get("choices") or []
                    if ch and ch[0].get("text"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append(now - last)
                        last = now
                        out += 1
                    if chunk.get("usage"):     # final usage chunk: authoritative token count
                        out = chunk["usage"].get("completion_tokens", out)
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
    if ok and out == 0:
        ok = False
        err = "empty response (0 tokens)"
    return {"ttft": ttft, "tpot": statistics.mean(itls) if itls else 0.0,
            "e2e": time.perf_counter() - t0, "out": out, "ok": ok, "err": err}


async def worker(client, base, model, max_tokens, stop, res):
    while time.perf_counter() < stop:
        res.append(await one_request(client, base, model, max_tokens))


async def run_level(base, model, concurrency, duration, max_tokens, timeout):
    res = []
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t = time.perf_counter()
        stop = t + duration
        tasks = [asyncio.create_task(worker(client, base, model, max_tokens, stop, res))
                 for _ in range(concurrency)]
        _, pending = await asyncio.wait(tasks, timeout=duration + timeout)
        for tk in pending:
            tk.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t

    ok = [r for r in res if r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    sample_err = next((r["err"] for r in res if not r["ok"] and r.get("err")), None)
    return {
        "concurrency": concurrency, "completed": len(res), "errors": len(res) - len(ok),
        "sample_error": sample_err, "wall_s": round(wall, 3),
        "req_per_s": round(len(ok) / wall, 3) if wall else 0.0,
        "throughput_tok_s": round(sum(r["out"] for r in ok) / wall, 2) if wall else 0.0,
        "ttft_p50": round(percentile(ttfts, 50), 4), "ttft_p95": round(percentile(ttfts, 95), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "tpot_p50": round(percentile([r["tpot"] for r in ok], 50), 4),
        "tpot_p95": round(percentile([r["tpot"] for r in ok], 95), 4),
        "e2e_p99": round(percentile([r["e2e"] for r in ok], 99), 4),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001", help="vLLM server base URL")
    ap.add_argument("--model", required=True, help="model id vLLM was served with")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--out", default="benchmarks/capstone_vllm.json")
    a = ap.parse_args()

    rows = []
    hdr = (f"{'conc':>5} {'req/s':>7} {'tok/s':>9} {'ttft_p50':>9} {'ttft_p95':>9} "
           f"{'tpot_p95':>9} {'e2e_p99':>9} {'err':>5}")
    print(f"\nBenchmarking vLLM {a.base}/v1/completions  (model={a.model}, max_tokens={a.max_tokens})")
    print("latencies in seconds\n")
    print(hdr)
    print("-" * len(hdr))
    for c in (int(x) for x in a.levels.split(",")):
        row = await run_level(a.base, a.model, c, a.duration, a.max_tokens, a.timeout)
        rows.append(row)
        print(f"{row['concurrency']:>5} {row['req_per_s']:>7.2f} {row['throughput_tok_s']:>9.1f} "
              f"{row['ttft_p50']:>9.3f} {row['ttft_p95']:>9.3f} {row['tpot_p95']:>9.4f} "
              f"{row['e2e_p99']:>9.3f} {row['errors']:>5}")
        if row["errors"]:
            print(f"      ^ {row['errors']} failed; e.g. {row['sample_error']}")

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
