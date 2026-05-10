"""
test_ordering_dolly.py — Real scheduling ordering test using Dolly 15K prompts

Structural mirror of test_queue_ordering.py but using real Dolly prompts instead
of hand-crafted SHORT/LONG fixtures. This removes the risk of synthetic prompts
not representing real workload distributions.

Test design:
  - 4 Short prompts from Dolly closed_qa category (factual Q&A, typically brief)
  - 4 Long  prompts from Dolly creative_writing category (stories, narratives)
  - Sends all 8 as a single burst to the scheduler
  - Expects: all Short prompts dispatched before any Long prompts

Why closed_qa → Short, creative_writing → Long:
  Dolly category distribution (from collect_dolly.py output):
    closed_qa       : S=500+ M=600+ L=2    (almost entirely Short/Medium)
    creative_writing: S=0    M=50   L=86   (Long-heavy — only Dolly category with L>50)

Scheduler must be running first:
    cd scheduler && go build -o clairvoyant ./cmd/main.go
    ./clairvoyant --model ../model/predictor.onnx --port 8080

Usage:
    python tests/test_ordering_dolly.py \\
        [--url http://localhost:8080] \\
        [--n-short 4] \\
        [--n-long 4]

Pass/Fail:
  PASS — all short_rank < all long_rank  (every Short dispatched before every Long)
  FAIL — any Long dispatched before a Short

Ranking accuracy reported: fraction of (short_i, long_j) pairs correctly ordered.
"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error

# ── Dolly prompt selection ────────────────────────────────────────────────────

def load_dolly_prompts(n_short: int, n_long: int):
    """
    Load real Dolly prompts.
    Returns (short_prompts: list[str], long_prompts: list[str])
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print("Loading Dolly 15K...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"  ✓ {len(ds):,} rows")

    short_prompts = []
    long_prompts  = []

    for row in ds:
        category = row.get("category", "")
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("context") or "").strip()
        response    = (row.get("response") or "").strip()

        if not instruction or not response:
            continue

        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContext: {context}"

        resp_tokens = len(response) // 4

        if category == "closed_qa" and resp_tokens < 200 and len(short_prompts) < n_short:
            short_prompts.append(prompt)

        if category == "creative_writing" and resp_tokens >= 800 and len(long_prompts) < n_long:
            long_prompts.append(prompt)

        if len(short_prompts) >= n_short and len(long_prompts) >= n_long:
            break

    if len(short_prompts) < n_short:
        print(f"WARNING: only found {len(short_prompts)} Short prompts "
              f"(needed {n_short}) in closed_qa. Relaxing to resp_tokens < 400.")
        for row in ds:
            if len(short_prompts) >= n_short:
                break
            category = row.get("category", "")
            instruction = (row.get("instruction") or "").strip()
            response    = (row.get("response") or "").strip()
            if category == "closed_qa" and len(response) // 4 < 400:
                if instruction not in short_prompts:
                    short_prompts.append(instruction)

    if len(long_prompts) < n_long:
        print(f"WARNING: only found {len(long_prompts)} Long prompts "
              f"(needed {n_long}) in creative_writing. Relaxing to resp_tokens >= 400.")
        for row in ds:
            if len(long_prompts) >= n_long:
                break
            category = row.get("category", "")
            instruction = (row.get("instruction") or "").strip()
            response    = (row.get("response") or "").strip()
            if category == "creative_writing" and len(response) // 4 >= 400:
                if instruction not in long_prompts:
                    long_prompts.append(instruction)

    return short_prompts[:n_short], long_prompts[:n_long]


# ── Scheduler interaction ─────────────────────────────────────────────────────

def send_prompt(url: str, prompt: str, label: str, results: list, idx: int):
    """
    Send prompt to scheduler, record dispatch_start_time.
    Appends (idx, label, dispatch_time) to results when response arrives.
    Uses /api/generate (Ollama-compatible endpoint).
    """
    payload = json.dumps({
        "model":  "gemma3:4b",
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.perf_counter() - t0
            results.append({
                "idx":     idx,
                "label":   label,
                "elapsed": elapsed,
            })
    except Exception as e:
        results.append({
            "idx":     idx,
            "label":   label,
            "elapsed": float("inf"),
            "error":   str(e),
        })


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test(base_url: str, n_short: int, n_long: int):
    short_prompts, long_prompts = load_dolly_prompts(n_short, n_long)

    print(f"\n  Short prompts ({len(short_prompts)}) from closed_qa:")
    for i, p in enumerate(short_prompts):
        print(f"    [{i}] {p[:80].strip()}...")

    print(f"\n  Long prompts ({len(long_prompts)}) from creative_writing:")
    for i, p in enumerate(long_prompts):
        print(f"    [{i}] {p[:80].strip()}...")

    print(f"\nSending {n_short + n_long} prompts as a burst to {base_url}...")

    results  = []
    threads  = []

    # Short prompts first in submission order, then Long
    # (A perfect SJF scheduler should still dispatch Short first regardless of order)
    all_prompts = (
        [(p, "SHORT") for p in short_prompts] +
        [(p, "LONG")  for p in long_prompts]
    )

    t_burst = time.perf_counter()

    for idx, (prompt, label) in enumerate(all_prompts):
        t = threading.Thread(
            target=send_prompt,
            args=(base_url, prompt, label, results, idx),
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300)

    # Sort by response completion time (proxy for dispatch order)
    results.sort(key=lambda r: r["elapsed"])

    print(f"\n{'─'*60}")
    print(f"  Dispatch order (by response completion time):")
    print(f"{'─'*60}")
    print(f"  {'rank':<6} {'label':<8} {'elapsed':>10}  {'note'}")
    print(f"  {'─'*52}")

    short_ranks = []
    long_ranks  = []

    for rank, r in enumerate(results, 1):
        err_note = f" ← ERROR: {r.get('error','')}" if "error" in r else ""
        print(f"  {rank:<6} {r['label']:<8} {r['elapsed']:>8.2f}s{err_note}")
        if r["label"] == "SHORT":
            short_ranks.append(rank)
        else:
            long_ranks.append(rank)

    # ── Ranking accuracy ──────────────────────────────────────────────────────
    correct_pairs = 0
    total_pairs   = 0
    for sr in short_ranks:
        for lr in long_ranks:
            total_pairs += 1
            if sr < lr:
                correct_pairs += 1

    ranking_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0

    # ── Pass/Fail ─────────────────────────────────────────────────────────────
    all_short_before_long = (
        max(short_ranks) < min(long_ranks)
        if short_ranks and long_ranks else False
    )

    print(f"\n{'═'*60}")
    print(f"  RESULTS")
    print(f"{'═'*60}")
    print(f"  Short completion ranks : {short_ranks}")
    print(f"  Long  completion ranks : {long_ranks}")
    print(f"  Ranking accuracy       : {ranking_acc*100:.1f}%  "
          f"({correct_pairs}/{total_pairs} pairs)")
    print(f"  Verdict                : {'✓ PASS' if all_short_before_long else '✗ FAIL'}")

    if not all_short_before_long and short_ranks and long_ranks:
        violating_long = [r for r in results if r["label"] == "LONG"
                          and r in results[:max(short_ranks)]]
        print(f"\n  Note: FAIL means at least one Long was dispatched before a Short.")
        print(f"  Check that the scheduler is running with the correct ONNX model.")

    return 0 if all_short_before_long else 1


def main():
    parser = argparse.ArgumentParser(
        description="Ordering test using real Dolly prompts (closed_qa vs creative_writing)"
    )
    parser.add_argument("--url",     default="http://localhost:8080")
    parser.add_argument("--n-short", type=int, default=4)
    parser.add_argument("--n-long",  type=int, default=4)
    args = parser.parse_args()

    sys.exit(run_test(args.url, args.n_short, args.n_long))


if __name__ == "__main__":
    main()
