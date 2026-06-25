#!/usr/bin/env python3
"""A persistent benchmark board for the ladder.

Each step registers its result into a shared board (benchmarks/board.json); the
`compare` command then plots every step on the board on one axis. This keeps the
comparison dynamic — add a step, re-run compare, and it shows up automatically.

Workflow:
    # after benchmarking a step, register its result file
    python scripts/board.py add --step 2 --label "Step 2 — KV cache" \
        --results benchmarks/step2_kv_cache.json
    python scripts/board.py add --step 3 --label "Step 3 — static batching" \
        --results benchmarks/step3_batching.json

    # plot everything currently on the board
    python scripts/board.py compare --out benchmarks/step_comparison.png

The x-axis metric (`concurrency` vs `batch_size`) is auto-detected per entry, so
steps measured different ways still overlay as "N-way parallelism vs throughput".
"""
import argparse
import json
import os

BOARD = "benchmarks/board.json"
# x-axis key -> how that step produced parallelism (for the legend / x-label)
X_KEYS = ("concurrency", "batch_size")
# per-step "seconds per request" key, in priority order (steps report it differently)
LATENCY_KEYS = ("median_s", "e2e_p99", "e2e_p50", "e2e")


def load_board(path):
    if not os.path.exists(path):
        return {"entries": []}
    with open(path) as f:
        return json.load(f)


def save_board(path, board):
    board["entries"].sort(key=lambda e: e["step"])
    with open(path, "w") as f:
        json.dump(board, f, indent=2)


def detect_x_key(results_path):
    with open(results_path) as f:
        rows = json.load(f)
    for row in rows:
        for k in X_KEYS:
            if k in row:
                return k
    raise SystemExit(f"Could not find an x-axis key ({X_KEYS}) in {results_path}")


def cmd_add(a):
    if not os.path.exists(a.results):
        raise SystemExit(f"Results file not found: {a.results}")
    x_key = a.x_key or detect_x_key(a.results)
    board = load_board(a.board)
    entry = {"step": a.step, "label": a.label, "results": a.results, "x_key": x_key}
    # upsert by step so re-running a step replaces its entry instead of duplicating
    board["entries"] = [e for e in board["entries"] if e["step"] != a.step]
    board["entries"].append(entry)
    save_board(a.board, board)
    print(f"Registered step {a.step} ({a.label}) -> {a.results} [x={x_key}]")
    print(f"Board now has {len(board['entries'])} step(s): "
          + ", ".join(str(e['step']) for e in sorted(board['entries'], key=lambda e: e['step'])))


def cmd_compare(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    board = load_board(a.board)
    entries = sorted(board["entries"], key=lambda e: e["step"])
    if not entries:
        raise SystemExit(f"Board {a.board} is empty — add a step first.")

    colors = ["#dc2626", "#2563eb", "#059669", "#9333ea", "#ea580c", "#0891b2"]
    markers = ["o", "s", "^", "D", "v", "P"]

    def first_present(row, keys):
        return next((k for k in keys if k in row), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(a.title, fontsize=13, fontweight="bold")
    all_x = set()
    for i, e in enumerate(entries):
        with open(e["results"]) as f:
            rows = [r for r in json.load(f) if a.metric in r]
        rows.sort(key=lambda r: r[e["x_key"]])
        xs = [r[e["x_key"]] for r in rows]
        all_x.update(xs)
        c, m = colors[i % len(colors)], markers[i % len(markers)]
        label = f"{e['label']}  ({e['x_key']})"

        # throughput panel
        axes[0].plot(xs, [r[a.metric] for r in rows], marker=m, linestyle="-", color=c, label=label)

        # latency panel — each step reports seconds/request under its own key
        lat_key = first_present(rows[0], LATENCY_KEYS) if rows else None
        if lat_key:
            axes[1].plot(xs, [r[lat_key] for r in rows], marker=m, linestyle="-", color=c,
                         label=f"{e['label']}  ({lat_key})")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(all_x))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("N  (concurrency or batch size)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(a.metric)
    axes[0].set_title("Throughput vs N")
    axes[1].set_ylabel("seconds / request")
    axes[1].set_title("Latency vs N")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=a.dpi)
    print(f"Wrote {a.out}  ({len(entries)} step(s))")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=BOARD, help="board manifest path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="register/replace a step's result on the board")
    p_add.add_argument("--step", type=int, required=True)
    p_add.add_argument("--label", required=True)
    p_add.add_argument("--results", required=True, help="benchmark result JSON for this step")
    p_add.add_argument("--x-key", choices=X_KEYS, default=None, help="override auto-detected x-axis key")
    p_add.set_defaults(func=cmd_add)

    p_cmp = sub.add_parser("compare", help="plot every step on the board")
    p_cmp.add_argument("--out", default="benchmarks/step_comparison.png")
    p_cmp.add_argument("--metric", default="throughput_tok_s", help="row key to plot on the y-axis")
    p_cmp.add_argument("--title", default="Step comparison — throughput vs N (Qwen2.5-0.5B)")
    p_cmp.add_argument("--dpi", type=int, default=120)
    p_cmp.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
