#!/usr/bin/env python3
"""Benchmark a streaming endpoint (app/api/inference.py).

The endpoint streams one Server-Sent-Event per token as `data: <token>\\n\\n`
(media_type text/plain) and ends when the greedy loop has emitted `max_length`
tokens. There's no [DONE] marker and no usage block — token count == number of
`data:` events == max_length.

Per request we capture the metrics that matter for interactive serving:
  * TTFT  — time to first token (request send -> first data: event)
  * TPOT  — time per output token (mean gap between successive token events)
  * e2e   — total wall time for the request

It sweeps concurrency levels closed-loop (each worker fires the next request as
soon as the previous finishes) and reports latency percentiles + token
throughput at each level — the saturation curve.

Usage:
    # auto-load the model on the server first, then sweep
    python scripts/benchmark_endpoint.py --model distilgpt2 --load

    python scripts/benchmark_endpoint.py --model distilgpt2 \
        --levels 1,2,4,8,16 --duration 20 --max-length 64
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

# Varied-length prompts; each request picks one at random. The length spread
# exercises ragged batching (different prompt lengths sharing a batch).
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
    """Linear-interpolated percentile over a list (stdlib only)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


async def load_model(base, model):
    """Call /load so the server has the model in memory before benchmarking."""
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{base}/load", params={"model_name": model})
        r.raise_for_status()
        print(f"[load] {r.json().get('message')}")


async def one_request(client, base, endpoint, model, max_length):
    """Stream a single endpoint call; capture TTFT, TPOT, e2e, token count."""
    params = {"model_name": model, "input_text": random.choice(PROMPTS), "max_length": max_length}
    t0 = time.perf_counter()
    ttft = None
    last = t0
    itls = []  # inter-token latencies (gaps after the first token)
    out = 0
    ok = False
    err = None
    try:
        async with client.stream("POST", f"{base}/{endpoint}", params=params) as r:
            ok = r.status_code == 200
            if not ok:
                await r.aread()
                err = f"HTTP {r.status_code}"
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        itls.append(now - last)
                    last = now
                    out += 1
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        # A dropped/reset connection counts as a failed request, not a fatal error —
        # record it so the sweep keeps going and the failure shows up in the report.
        ok = False
        err = f"{type(e).__name__}: {e}"
    return {
        "ttft": ttft,
        "tpot": statistics.mean(itls) if itls else 0.0,
        "e2e": time.perf_counter() - t0,
        "out": out,
        "ok": ok,
        "err": err,
    }


async def worker(client, base, endpoint, model, max_length, stop, res):
    while time.perf_counter() < stop:
        res.append(await one_request(client, base, endpoint, model, max_length))


async def run_level(base, endpoint, model, concurrency, duration, max_length):
    res = []
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t = time.perf_counter()
        stop = t + duration
        await asyncio.gather(
            *[worker(client, base, endpoint, model, max_length, stop, res) for _ in range(concurrency)]
        )
        wall = time.perf_counter() - t

    ok = [r for r in res if r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    sample_err = next((r["err"] for r in res if not r["ok"] and r.get("err")), None)
    return {
        "concurrency": concurrency,
        "completed": len(res),
        "errors": len(res) - len(ok),
        "sample_error": sample_err,
        "wall_s": round(wall, 3),
        "req_per_s": round(len(ok) / wall, 3) if wall else 0.0,
        "throughput_tok_s": round(sum(r["out"] for r in ok) / wall, 2) if wall else 0.0,
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p95": round(percentile(ttfts, 95), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "tpot_p50": round(percentile([r["tpot"] for r in ok], 50), 4),
        "tpot_p95": round(percentile([r["tpot"] for r in ok], 95), 4),
        "e2e_p99": round(percentile([r["e2e"] for r in ok], 99), 4),
    }


async def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="http://localhost:8000", help="server base URL")
    ap.add_argument("--endpoint", default="generate-stream", help="endpoint to benchmark")
    ap.add_argument("--model", required=True, help="model_name passed to the endpoint")
    ap.add_argument("--load", action="store_true", help="call /load before sweeping")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--duration", type=float, default=20, help="seconds per concurrency level")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128,256", help="comma-separated concurrency levels")
    ap.add_argument("--out", default="bench_endpoint_results.json")
    a = ap.parse_args()

    if a.load:
        await load_model(a.base, a.model)

    url = f"{a.base}/{a.endpoint}"
    rows = []
    hdr = (
        f"{'conc':>5} {'req/s':>7} {'tok/s':>9} "
        f"{'ttft_p50':>9} {'ttft_p95':>9} {'ttft_p99':>9} "
        f"{'tpot_p95':>9} {'e2e_p99':>9} {'err':>4}"
    )
    print(f"\nBenchmarking {url}  (model={a.model}, max_length={a.max_length})")
    print("latencies in seconds\n")
    print(hdr)
    print("-" * len(hdr))
    for c in (int(x) for x in a.levels.split(",")):
        row = await run_level(a.base, a.endpoint, a.model, c, a.duration, a.max_length)
        rows.append(row)
        print(
            f"{row['concurrency']:>5} {row['req_per_s']:>7.2f} {row['throughput_tok_s']:>9.1f} "
            f"{row['ttft_p50']:>9.3f} {row['ttft_p95']:>9.3f} {row['ttft_p99']:>9.3f} "
            f"{row['tpot_p95']:>9.4f} {row['e2e_p99']:>9.3f} {row['errors']:>4}"
        )
        if row["errors"]:
            print(f"      ^ {row['errors']} failed; e.g. {row['sample_error']}")

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
