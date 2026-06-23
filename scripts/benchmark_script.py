# bench.py
import asyncio, time, json, statistics, argparse
import httpx, numpy as np

PROMPT = "Explain how paged attention reduces KV cache fragmentation."  # fixed for fair comparison

async def one_request(client, url, model, max_tokens):
    payload = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; last = t0; itls = []; out = 0
    async with client.stream("POST", url, json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            chunk = json.loads(data)
            ch = chunk.get("choices", [])
            if ch and ch[0].get("text"):
                now = time.perf_counter()
                if ttft is None: ttft = now - t0
                else: itls.append(now - last)
                last = now; out += 1
            if chunk.get("usage"): out = chunk["usage"]["completion_tokens"]
    return {"ttft": ttft, "e2e": time.perf_counter() - t0, "out": out,
            "tpot": statistics.mean(itls) if itls else 0.0}

async def worker(client, url, model, mt, stop, res):
    while time.perf_counter() < stop:
        res.append(await one_request(client, url, model, mt))

async def run_level(url, model, c, dur, mt):
    res = []
    limits = httpx.Limits(max_connections=c * 2, max_keepalive_connections=c * 2)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t = time.perf_counter(); stop = t + dur
        await asyncio.gather(*[worker(client, url, model, mt, stop, res) for _ in range(c)])
        wall = time.perf_counter() - t
    pct = lambda xs, p: float(np.percentile(xs, p)) if xs else 0.0
    ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
    return {"concurrency": c, "completed": len(res),
            "throughput_tok_s": sum(r["out"] for r in res) / wall,
            "req_per_s": len(res) / wall,
            "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95), "ttft_p99": pct(ttfts, 99),
            "tpot_p95": pct([r["tpot"] for r in res], 95),
            "e2e_p99": pct([r["e2e"] for r in res], 99)}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/generate/")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    a = ap.parse_args()
    rows = []
    for c in (int(x) for x in a.levels.split(",")):
        row = await run_level(a.url, a.model, c, a.duration, a.max_tokens)
        rows.append(row); print(json.dumps(row))
    json.dump(rows, open("bench_results.json", "w"), indent=2)

asyncio.run(main())