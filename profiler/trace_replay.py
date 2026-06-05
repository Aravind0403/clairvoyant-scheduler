"""
trace_replay.py — Realistic workload replay with TTFT decomposition.

Sends prompts at Poisson arrival intervals and measures per-request:
  - TTFT      : request sent → first streaming chunk received
  - End-to-end: request sent → last chunk received
  - Queue wait : arrival_time → first_token_time  (≈ TTFT on serial backend)

Usage — FCFS condition (direct Ollama):
    python profiler/trace_replay.py \\
        --endpoint http://localhost:11434/v1/chat/completions \\
        --model gemma3:4b --rho 0.8 --n 150 \\
        --label fcfs --save results/trace_replay_fcfs_rho08.json

Usage — SJF condition (via Clairvoyant):
    python profiler/trace_replay.py \\
        --endpoint http://localhost:8080/v1/chat/completions \\
        --model gemma3:4b --rho 0.8 --n 150 \\
        --label sjf --save results/trace_replay_sjf_rho08.json

Prompt source:
    Expects data/lmsys_labeled.csv with columns:
        prompt, actual_output_tokens  (+ feature columns)
    If not present, falls back to built-in probe prompts.

E[S] calibration:
    Default E[S]=6.2s (RTX 4090, Gemma3:4b, 50/50 short/long mix).
    Override with --es-seconds if running on different hardware.
"""

import argparse
import csv
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# ── Fallback prompts if CSV not available ────────────────────────────────────

FALLBACK_PROMPTS = [
    ("What is the capital of France?",                                          "SHORT"),
    ("Define recursion in one sentence.",                                       "SHORT"),
    ("What does HTTP stand for?",                                               "SHORT"),
    ("Name three sorting algorithms.",                                          "SHORT"),
    ("What is a neural network?",                                               "SHORT"),
    ("Explain the difference between TCP and UDP.",                             "SHORT"),
    ("What is the Pythagorean theorem?",                                        "SHORT"),
    ("What year did World War II end?",                                         "SHORT"),
    ("Write a Python function to reverse a string.",                            "MEDIUM"),
    ("Explain how garbage collection works in Java.",                           "MEDIUM"),
    ("What are the SOLID principles in software engineering?",                  "MEDIUM"),
    ("Describe the difference between SQL and NoSQL databases.",                "MEDIUM"),
    ("Write a short story about a robot learning to paint.",                    "LONG"),
    ("Explain the history of the internet from ARPANET to today.",              "LONG"),
    ("Write a Python program to parse Apache logs and report top 10 IPs.",     "LONG"),
    ("Create a React component for a todo list with add, delete, and filter.",  "LONG"),
]

# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    idx:              int
    prompt_class:     str        # SHORT / MEDIUM / LONG
    arrival_time:     float      # wall-clock when dispatcher scheduled this req
    send_time:        float = 0.0
    first_token_time: float = 0.0
    completion_time:  float = 0.0
    error:            str   = ""

    @property
    def queue_wait_s(self) -> Optional[float]:
        if self.first_token_time == 0.0: return None
        return self.first_token_time - self.arrival_time

    @property
    def ttft_s(self) -> Optional[float]:
        if self.first_token_time == 0.0: return None
        return self.first_token_time - self.send_time

    @property
    def end_to_end_s(self) -> Optional[float]:
        if self.completion_time == 0.0: return None
        return self.completion_time - self.arrival_time

    @property
    def ok(self) -> bool:
        return not self.error and self.completion_time > 0.0


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_prompts(csv_path: str, n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Returns list of (prompt, class) tuples sampled from CSV or fallback."""
    p = Path(csv_path)
    if not p.exists():
        print(f"  [warn] {csv_path} not found — using built-in fallback prompts")
        rng = random.Random(seed)
        pool = FALLBACK_PROMPTS * ((n // len(FALLBACK_PROMPTS)) + 2)
        return rng.sample(pool, min(n, len(pool)))

    rows = []
    with open(p) as f:
        for row in csv.DictReader(f):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                tokens = int(row.get("actual_output_tokens", 0))
            except ValueError:
                tokens = 0
            if tokens < 200:
                cls = "SHORT"
            elif tokens < 800:
                cls = "MEDIUM"
            else:
                cls = "LONG"
            rows.append((prompt, cls))

    rng = random.Random(seed)
    rng.shuffle(rows)
    sampled = rows[:n]
    counts = {c: sum(1 for _, c2 in sampled if c2 == c) for c in ("SHORT", "MEDIUM", "LONG")}
    print(f"  Loaded {len(sampled)} prompts — SHORT:{counts['SHORT']} "
          f"MEDIUM:{counts['MEDIUM']} LONG:{counts['LONG']}")
    return sampled


# ── HTTP worker — streaming ───────────────────────────────────────────────────

def send_streaming(endpoint: str, model: str,
                   prompt: str, result: RequestResult) -> None:
    """Send request in streaming mode; record first-token and completion times."""
    payload = json.dumps({
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   True,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result.send_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            first = True
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                # Check for actual content token
                choices = obj.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if first:
                        result.first_token_time = time.perf_counter()
                        first = False
            result.completion_time = time.perf_counter()
    except urllib.error.HTTPError as e:
        result.error = f"HTTP {e.code}: {e.reason}"
        result.completion_time = time.perf_counter()
    except Exception as e:
        result.error = str(e)
        result.completion_time = time.perf_counter()


# ── Poisson dispatcher ────────────────────────────────────────────────────────

def run_replay(endpoint: str, model: str,
               prompts: list[tuple[str, str]],
               lam: float) -> list[RequestResult]:
    """
    Fire requests at Poisson(λ) arrival rate.
    Each request runs in its own daemon thread.
    Returns results list populated by workers.
    """
    results: list[RequestResult] = []
    threads: list[threading.Thread] = []

    t0 = time.perf_counter()

    for idx, (prompt, cls) in enumerate(prompts):
        arrival = time.perf_counter()
        result = RequestResult(idx=idx, prompt_class=cls, arrival_time=arrival)
        results.append(result)

        t = threading.Thread(
            target=send_streaming,
            args=(endpoint, model, prompt, result),
            daemon=True,
        )
        threads.append(t)
        t.start()

        if idx < len(prompts) - 1:
            inter_arrival = random.expovariate(lam)
            time.sleep(inter_arrival)

    for t in threads:
        t.join(timeout=600)

    elapsed = time.perf_counter() - t0
    print(f"  Wall time: {elapsed:.1f}s  ({len(prompts)} requests)")
    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def print_summary(results: list[RequestResult], label: str) -> dict:
    ok = [r for r in results if r.ok]
    print(f"\n{'═'*65}")
    print(f"  {label.upper()}  —  {len(ok)}/{len(results)} successful")
    print(f"{'═'*65}")

    summary = {"label": label, "n_ok": len(ok), "n_total": len(results)}

    for cls in ("SHORT", "LONG", "ALL"):
        subset = [r for r in ok if cls == "ALL" or r.prompt_class == cls]
        if not subset:
            continue

        ttft    = [r.ttft_s        for r in subset if r.ttft_s        is not None]
        qw      = [r.queue_wait_s  for r in subset if r.queue_wait_s  is not None]
        e2e     = [r.end_to_end_s  for r in subset if r.end_to_end_s  is not None]

        print(f"\n  [{cls}]  n={len(subset)}")
        print(f"  {'Metric':<20} {'P50':>8} {'P95':>8} {'Mean':>8}")
        print(f"  {'-'*48}")
        for name, vals in [("TTFT (s)", ttft), ("Queue Wait (s)", qw), ("End-to-End (s)", e2e)]:
            if vals:
                p50  = percentile(vals, 50)
                p95  = percentile(vals, 95)
                mean = statistics.mean(vals)
                print(f"  {name:<20} {p50:>8.2f} {p95:>8.2f} {mean:>8.2f}")
                summary[f"{cls}_{name.split()[0].lower()}_p50"]  = round(p50,  3)
                summary[f"{cls}_{name.split()[0].lower()}_p95"]  = round(p95,  3)
                summary[f"{cls}_{name.split()[0].lower()}_mean"] = round(mean, 3)

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Poisson trace replay with TTFT decomposition"
    )
    parser.add_argument("--endpoint",   default="http://localhost:11434/v1/chat/completions")
    parser.add_argument("--model",      default="gemma3:4b")
    parser.add_argument("--rho",        type=float, default=0.8,
                        help="Target queue utilisation (0 < ρ < 1)")
    parser.add_argument("--es-seconds", type=float, default=6.2,
                        help="E[S] in seconds (default: 6.2s for RTX 4090 Gemma3:4b)")
    parser.add_argument("--n",          type=int,   default=150,
                        help="Number of requests to send")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--label",      default="run",
                        help="Condition label (e.g. fcfs, sjf)")
    parser.add_argument("--data",       default="data/lmsys_labeled.csv")
    parser.add_argument("--save",       default="results/trace_replay.json")
    args = parser.parse_args()

    lam = args.rho / args.es_seconds
    print(f"\nTrace Replay — {args.label.upper()}")
    print(f"  Endpoint  : {args.endpoint}")
    print(f"  Model     : {args.model}")
    print(f"  ρ={args.rho}  E[S]={args.es_seconds}s  λ={lam:.4f} req/s  "
          f"(inter-arrival mean: {1/lam:.1f}s)")
    print(f"  n={args.n}  seed={args.seed}")

    random.seed(args.seed)
    prompts = load_prompts(args.data, args.n, seed=args.seed)

    print(f"\nStarting replay...")
    results = run_replay(args.endpoint, args.model, prompts, lam)

    summary = print_summary(results, args.label)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "label":      args.label,
        "endpoint":   args.endpoint,
        "model":      args.model,
        "rho":        args.rho,
        "es_seconds": args.es_seconds,
        "lambda":     round(lam, 6),
        "n":          args.n,
        "seed":       args.seed,
        "summary":    summary,
        "requests": [
            {
                "idx":              r.idx,
                "prompt_class":     r.prompt_class,
                "arrival_time":     round(r.arrival_time,     4),
                "send_time":        round(r.send_time,        4),
                "first_token_time": round(r.first_token_time, 4),
                "completion_time":  round(r.completion_time,  4),
                "ttft_s":           round(r.ttft_s,           4) if r.ttft_s       is not None else None,
                "queue_wait_s":     round(r.queue_wait_s,     4) if r.queue_wait_s is not None else None,
                "end_to_end_s":     round(r.end_to_end_s,     4) if r.end_to_end_s is not None else None,
                "error":            r.error,
            }
            for r in results
        ],
    }

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {save_path}")


if __name__ == "__main__":
    main()
