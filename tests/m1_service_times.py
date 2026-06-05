"""
m1_service_times.py — Measure raw Ollama service times for Figure 1 / Table 1.

Sends short (closed_qa) and long (creative_writing) Dolly prompts directly to
Ollama (bypassing Clairvoyant) and records wall-clock generation time per request.

Run with Ollama running locally (no scheduler needed):
    python tests/m1_service_times.py [--n 3] [--model gemma3:4b]

Output saved to: results/m1_service_times.json
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


OLLAMA_URL = "http://localhost:11434"


def load_dolly_prompts(n: int):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print("Loading Dolly 15K...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    short_prompts, long_prompts = [], []

    for row in ds:
        category    = row.get("category", "")
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("context") or "").strip()
        response    = (row.get("response") or "").strip()

        if not instruction or not response:
            continue

        prompt      = f"{instruction}\n\nContext: {context}" if context else instruction
        resp_tokens = len(response) // 4

        if category == "closed_qa" and resp_tokens < 200 and len(short_prompts) < n:
            short_prompts.append(prompt)

        if category == "creative_writing" and resp_tokens >= 800 and len(long_prompts) < n:
            long_prompts.append(prompt)

        if len(short_prompts) >= n and len(long_prompts) >= n:
            break

    print(f"  ✓ {len(short_prompts)} short prompts, {len(long_prompts)} long prompts")
    return short_prompts[:n], long_prompts[:n]


def time_request(prompt: str, model: str) -> float:
    """Send prompt directly to Ollama, return wall-clock seconds."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        _ = resp.read()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=3, help="Requests per class")
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    short_prompts, long_prompts = load_dolly_prompts(args.n)

    records = []

    print(f"\nTiming {args.n} SHORT requests (closed_qa) → Ollama direct...")
    for i, prompt in enumerate(short_prompts):
        print(f"  [{i+1}/{args.n}] {prompt[:60].strip()}...")
        elapsed = time_request(prompt, args.model)
        print(f"        → {elapsed:.2f}s")
        records.append({"class": "SHORT", "elapsed_s": round(elapsed, 3), "prompt_preview": prompt[:80]})

    print(f"\nTiming {args.n} LONG requests (creative_writing) → Ollama direct...")
    for i, prompt in enumerate(long_prompts):
        print(f"  [{i+1}/{args.n}] {prompt[:60].strip()}...")
        elapsed = time_request(prompt, args.model)
        print(f"        → {elapsed:.2f}s")
        records.append({"class": "LONG", "elapsed_s": round(elapsed, 3), "prompt_preview": prompt[:80]})

    short_times = [r["elapsed_s"] for r in records if r["class"] == "SHORT"]
    long_times  = [r["elapsed_s"] for r in records if r["class"] == "LONG"]

    avg_short = sum(short_times) / len(short_times)
    avg_long  = sum(long_times)  / len(long_times)
    gap       = avg_long - avg_short

    print(f"\n{'═'*55}")
    print(f"  SHORT  times : {[f'{t:.1f}s' for t in short_times]}  avg={avg_short:.1f}s")
    print(f"  LONG   times : {[f'{t:.1f}s' for t in long_times]}  avg={avg_long:.1f}s")
    print(f"  Gap (long−short) : {gap:.1f}s  ← use this for Figure 1")
    print(f"{'═'*55}")

    out = {
        "model":     args.model,
        "backend":   "Ollama direct (no scheduler)",
        "hardware":  "Apple M1 (set manually if needed)",
        "n_per_class": args.n,
        "records":   records,
        "summary": {
            "avg_short_s": round(avg_short, 2),
            "avg_long_s":  round(avg_long,  2),
            "gap_s":       round(gap,        2),
        }
    }

    out_path = Path(__file__).parent.parent / "results" / "m1_service_times.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
