#!/usr/bin/env python3
"""Profile the /generate-continuous-batch endpoint with cProfile and compare runs.

Only the request traffic is profiled. Startup, imports and model loading are not:
the script starts its own server, loads the model, turns the profiler on, sends the
requests, and turns the profiler off.

Each run saves two files in benchmarks/:
  * profile_<name>.prof  the raw cProfile data (open with snakeviz or pstats)
  * profile_<name>.json  a summary: time per bucket (decode, prefill, sampling, ...)

Usage:
    # profile a run (uses port 8002, so a server on 8000 can keep running)
    python scripts/profile_cprofile.py run --name after_my_change

    # compare it with the baseline
    python scripts/profile_cprofile.py compare benchmarks/profile_cb_baseline.json \
        benchmarks/profile_after_my_change.json

    # rebuild a summary from a .prof file (for example after adding a bucket)
    python scripts/profile_cprofile.py summarize benchmarks/profile_cb_baseline.prof

To track a new function, add a line to BUCKETS. Old .prof files can be summarized
again, so the new bucket also shows up for older runs.
"""
import argparse
import asyncio
import cProfile
import json
import os
import pstats
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

import httpx

from profile_service import PROMPTS  # same prompts as the engine-timing profiler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each bucket picks one function from the profile and reports its cumulative time.
# "file" is matched against the end of the path ("~" means a C built-in).
# "caller" counts only the calls made from that function.
BUCKETS = [
    {"name": "engine_loop", "file": "app/batching/engine.py", "func": "_run"},
    {"name": "step", "file": "app/batching/engine.py", "func": "_step"},
    {"name": "forward", "file": "app/model.py", "func": "forward"},
    {"name": "decode", "file": "app/batching/engine.py", "func": "_decode"},
    {"name": "prefill", "file": "app/batching/engine.py", "func": "_prefill"},
    {"name": "sampling", "file": "app/api/utils.py", "func": "sample_next_token_batched"},
    {"name": "step_tensor_copy", "file": "~", "func": "<built-in method torch.tensor>",
     "caller": ("app/batching/engine.py", "_step")},
    {"name": "step_tolist", "file": "~", "func": "<method 'tolist' of 'torch._C.TensorBase' objects>",
     "caller": ("app/batching/engine.py", "_step")},
    {"name": "splice", "file": "app/batching/engine.py", "func": "_splice"},
    {"name": "evict", "file": "app/batching/engine.py", "func": "_evict_finished"},
    {"name": "trim_left_pad", "file": "app/batching/engine.py", "func": "_trim_left_pad"},
    {"name": "position_ids", "file": "app/batching/engine.py", "func": "_position_ids"},
    {"name": "submit", "file": "app/batching/engine.py", "func": "submit"},
    {"name": "endpoint_handler", "file": "app/api/inference.py", "func": "generate_text_continuous_batch"},
    {"name": "stream_generator", "file": "app/api/inference.py", "func": "stream_generator"},
]


# ---------------------------------------------------------------- server side

def cmd_serve(a):
    """Run the app with /_prof/start and /_prof/stop routes. Called by `run`."""
    sys.path.insert(0, REPO)
    import uvicorn
    from main import app

    prof = cProfile.Profile()

    @app.post("/_prof/start")
    async def prof_start():
        prof.enable()
        return {"ok": True}

    @app.post("/_prof/stop")
    async def prof_stop():
        prof.disable()
        prof.dump_stats(a.out)
        return {"saved": a.out}

    uvicorn.run(app, host="localhost", port=a.port, log_level="warning")


# ---------------------------------------------------------------- client side

def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_ready(base, server, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError("profiling server exited early, check the server log")
        try:
            if httpx.get(f"{base}/", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError("profiling server did not start in time")


async def one(client, base, model, prompt, max_length):
    params = {"model_name": model, "input_text": prompt, "max_length": max_length}
    async with client.stream("POST", f"{base}/generate-continuous-batch", params=params) as r:
        r.raise_for_status()
        async for _ in r.aiter_lines():  # read the stream to the end
            pass


async def drive(base, a):
    """Load the model, warm up, then send the profiled requests. Returns wall time."""
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{base}/load", params={
            "model_name": a.model, "batch_size": a.batch_size, "max_length": a.max_length})
        r.raise_for_status()

        # warm-up requests: CUDA and cuBLAS set-up costs land here, not in the profile
        for i in range(a.warmup):
            await one(client, base, a.model, PROMPTS[i % len(PROMPTS)], a.max_length)

        sem = asyncio.Semaphore(a.concurrency)

        async def worker(i):
            async with sem:
                await one(client, base, a.model, PROMPTS[i % len(PROMPTS)], a.max_length)

        (await client.post(f"{base}/_prof/start")).raise_for_status()
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(a.n)])
        wall = time.perf_counter() - t0
        (await client.post(f"{base}/_prof/stop")).raise_for_status()
    return wall


def cmd_run(a):
    if port_in_use(a.port):
        sys.exit(f"port {a.port} is already in use, pick another one with --port")

    os.makedirs(a.out_dir, exist_ok=True)
    prof_path = os.path.join(a.out_dir, f"profile_{a.name}.prof")
    log_path = os.path.join(a.out_dir, f"profile_{a.name}.server.log")  # *.log is gitignored
    base = f"http://localhost:{a.port}"

    print(f"[run] starting profiling server on port {a.port} (log: {log_path})")
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "serve", "--out", prof_path, "--port", str(a.port)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(base, server)
            print(f"[run] loading {a.model}, {a.warmup} warm-up request(s), then {a.n} profiled requests "
                  f"({a.concurrency} at a time, max_length={a.max_length})")
            wall = asyncio.run(drive(base, a))
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()

    config = {"model": a.model, "n": a.n, "concurrency": a.concurrency, "max_length": a.max_length,
              "batch_size": a.batch_size, "warmup": a.warmup, "wall_s": round(wall, 3)}
    summary = summarize(prof_path, a.name, config)
    write_summary(summary, os.path.join(a.out_dir, f"profile_{a.name}.json"))
    print_table([summary])


# ---------------------------------------------------------------- summary

def match(key, file, func):
    filename, _, funcname = key
    if funcname != func:
        return False
    return filename == "~" if file == "~" else filename.endswith(file)


def bucket_stats(stats, b):
    """Return (calls, cumulative seconds) for one bucket."""
    calls, cum = 0, 0.0
    for key, (_, nc, _, ct, callers) in stats.items():
        if not match(key, b["file"], b["func"]):
            continue
        if "caller" not in b:
            calls, cum = calls + nc, cum + ct
            continue
        for ckey, cvals in callers.items():
            if match(ckey, *b["caller"]):
                calls, cum = calls + cvals[1], cum + cvals[3]
    return calls, cum


def summarize(prof_path, name, config=None):
    st = pstats.Stats(prof_path)
    total = st.total_tt
    buckets = {}
    for b in BUCKETS:
        calls, cum = bucket_stats(st.stats, b)
        buckets[b["name"]] = {
            "s": round(cum, 4),
            "calls": calls,
            "ms_per_call": round(cum / calls * 1000, 4) if calls else None,
            "share_pct": round(cum / total * 100, 2) if total else None,
        }
    top = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:20]
    return {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "prof_file": os.path.relpath(prof_path, REPO),
        "config": config or {},
        "total_s": round(total, 4),
        "buckets": buckets,
        "top_tottime": [
            {"func": f"{os.path.basename(k[0])}:{k[1]}({k[2]})", "calls": v[1],
             "tottime_s": round(v[2], 4), "cumtime_s": round(v[3], 4)}
            for k, v in top
        ],
    }


def write_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {path}")


def cmd_summarize(a):
    out = a.out or os.path.splitext(a.prof)[0] + ".json"
    name = a.name or os.path.splitext(os.path.basename(a.prof))[0].removeprefix("profile_")
    config = {}
    if os.path.exists(out):  # keep the run config from an earlier summary
        with open(out) as f:
            config = json.load(f).get("config", {})
    summary = summarize(a.prof, name, config)
    write_summary(summary, out)
    print_table([summary])


# ---------------------------------------------------------------- compare

def pct(old, new):
    if not old or new is None:
        return ""
    return f"{(new - old) / old * 100:+.1f}%"


def print_table(summaries):
    """One summary: print it. Two summaries: print them side by side with the change."""
    names = [s["name"] for s in summaries]
    print()
    for s in summaries:
        c = s["config"]
        print(f"  {s['name']}: total {s['total_s']:.2f} s, wall {c.get('wall_s', '?')} s, "
              f"n={c.get('n', '?')}, concurrency={c.get('concurrency', '?')}, "
              f"max_length={c.get('max_length', '?')}, batch_size={c.get('batch_size', '?')}")
    print()

    if len(summaries) == 1:
        s = summaries[0]
        print(f"{'bucket':<18}{'seconds':>10}{'share':>9}{'calls':>9}{'ms/call':>11}")
        print("-" * 57)
        for name, b in s["buckets"].items():
            mpc = f"{b['ms_per_call']:.3f}" if b["ms_per_call"] is not None else "-"
            print(f"{name:<18}{b['s']:>10.3f}{b['share_pct']:>8.1f}%{b['calls']:>9}{mpc:>11}")
        return

    a, b = summaries
    print(f"{'bucket':<18}{'s (A)':>9}{'s (B)':>9}{'change':>9}"
          f"{'ms/call (A)':>13}{'ms/call (B)':>13}{'change':>9}{'share A→B':>15}")
    print("-" * 95)
    keys = list(a["buckets"]) + [k for k in b["buckets"] if k not in a["buckets"]]
    for k in keys:
        x, y = a["buckets"].get(k), b["buckets"].get(k)
        if x is None or y is None:
            print(f"{k:<18}  (only in {'A' if y is None else 'B'})")
            continue
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        print(f"{k:<18}{x['s']:>9.3f}{y['s']:>9.3f}{pct(x['s'], y['s']):>9}"
              f"{fmt(x['ms_per_call']):>13}{fmt(y['ms_per_call']):>13}"
              f"{pct(x['ms_per_call'], y['ms_per_call']):>9}"
              f"{x['share_pct']:>7.1f}% → {y['share_pct']:.1f}%")
    wa, wb = a["config"].get("wall_s"), b["config"].get("wall_s")
    if wa and wb:
        print("-" * 95)
        print(f"{'wall time':<18}{wa:>9.3f}{wb:>9.3f}{pct(wa, wb):>9}")
    print(f"\nA = {names[0]}, B = {names[1]}. Compare ms/call when the runs differ in size.")


def cmd_compare(a):
    summaries = []
    for path in (a.base, a.new):
        with open(path) as f:
            summaries.append(json.load(f))
    print_table(summaries)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="profile the endpoint and save .prof + .json")
    p.add_argument("--name", required=True, help="run name, used in the file names")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--n", type=int, default=100, help="number of profiled requests")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1, help="requests sent before profiling starts")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--out-dir", default=os.path.join(REPO, "benchmarks"))
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("summarize", help="rebuild the .json summary from a .prof file")
    p.add_argument("prof")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("compare", help="compare two .json summaries")
    p.add_argument("base", help="A: the reference run, for example the baseline")
    p.add_argument("new", help="B: the run to compare against it")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("serve", help=argparse.SUPPRESS)  # internal, started by `run`
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8002)
    p.set_defaults(fn=cmd_serve)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
