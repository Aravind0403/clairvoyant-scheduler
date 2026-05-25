"""
FCFS vs SJF end-to-end latency benchmark for Clairvoyant.

Sends a mixed workload (short + long prompts) concurrently to two endpoints:
  - FCFS: direct to Ollama (no proxy)
  - SJF:  through Clairvoyant proxy

Reports P50 / P95 / P99 for short-class requests per condition.

Usage
-----
  # Run both conditions back to back (5 runs each):
  python benchmark.py --model gemma3:4b --runs 5

  # Single condition:
  python benchmark.py --condition fcfs --model gemma3:4b --runs 5

Output
------
  benchmark_results.csv  — one row per request per run per condition
  benchmark_summary.csv  — P50/P95/P99 table for the paper
"""

import argparse
import csv
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ── Endpoints ─────────────────────────────────────────────────────────────────

FCFS_URL = "http://localhost:11434/v1/chat/completions"   # direct Ollama
SJF_URL  = "http://localhost:8080/v1/chat/completions"    # Clairvoyant proxy

# ── Prompt bank (50 short + 50 long) ─────────────────────────────────────────

SHORT_PROMPTS = [
    "What is the capital of France?",
    "Name three primary colours.",
    "What does CPU stand for?",
    "Who wrote Romeo and Juliet?",
    "What is 17 multiplied by 13?",
    "Define the term 'latency' in networking.",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in the solar system.",
    "What language is Django written in?",
    "What does HTTP stand for?",
    "What is a REST API?",
    "Name two sorting algorithms.",
    "What is the time complexity of binary search?",
    "What does RAM stand for?",
    "What is the difference between TCP and UDP?",
    "Who invented the telephone?",
    "What is a pointer in C?",
    "Define 'idempotent' in the context of HTTP methods.",
    "What is a foreign key in SQL?",
    "What does JSON stand for?",
    "What is a hash function?",
    "Name three NoSQL databases.",
    "What is the purpose of a load balancer?",
    "What is garbage collection in programming?",
    "What does CI/CD stand for?",
    "What is a mutex?",
    "Define 'throughput' in systems.",
    "What is the OSI model?",
    "What is a deadlock?",
    "What does DNS stand for?",
    "What is a virtual machine?",
    "What is MapReduce?",
    "Name two message queue systems.",
    "What is Kubernetes used for?",
    "What is a container in software?",
    "What is the purpose of an index in a database?",
    "What is ACID in databases?",
    "What is a race condition?",
    "What does API stand for?",
    "What is the difference between a process and a thread?",
    "What is a CDN?",
    "What is tail latency?",
    "What is a heap data structure?",
    "What is a bloom filter?",
    "What does TLS stand for?",
    "What is consistent hashing?",
    "What is a circuit breaker pattern?",
    "Name two observability tools.",
    "What is sharding in databases?",
    "What is eventual consistency?",
]

LONG_PROMPTS = [
    "Write a Python function to implement merge sort with detailed comments explaining each step.",
    "Explain the internals of how a B-tree index works in a relational database, with an example.",
    "Write a complete REST API in Python using FastAPI for a user management system with CRUD operations.",
    "Implement a distributed rate limiter using Redis and explain the algorithm in detail.",
    "Write a Python script that monitors system CPU and memory usage and sends alerts via email.",
    "Explain the Raft consensus algorithm step by step with a concrete example scenario.",
    "Write a complete implementation of a thread-safe LRU cache in Python.",
    "Design a URL shortener system and explain the architecture, database schema, and tradeoffs.",
    "Write a Python implementation of Dijkstra's algorithm with a detailed explanation.",
    "Explain how PostgreSQL handles MVCC (multi-version concurrency control) internally.",
    "Write a complete Docker Compose file for a microservices application with Nginx, FastAPI, and PostgreSQL.",
    "Implement a producer-consumer queue in Python using threading and explain the synchronisation.",
    "Write a Python class that implements a binary search tree with insert, delete, and traversal.",
    "Explain how a GPU processes a CUDA kernel, including thread blocks, warps, and memory hierarchy.",
    "Write a complete CI/CD pipeline configuration for a Python project using GitHub Actions.",
    "Design a real-time chat application architecture and explain the key components.",
    "Write a Python implementation of consistent hashing with virtual nodes.",
    "Explain how transformers compute self-attention mathematically, step by step.",
    "Write a Kubernetes deployment manifest for a high-availability web application.",
    "Implement a pub-sub event system in Python and explain when you would use it.",
    "Write a complete implementation of the producer-consumer pattern using asyncio in Python.",
    "Explain the internals of Linux memory management including virtual memory and page tables.",
    "Write a Python script to crawl a website recursively and extract all internal links.",
    "Design a distributed key-value store and explain replication and fault tolerance.",
    "Write a complete implementation of a bloom filter in Python with false positive analysis.",
    "Explain how TCP handles congestion control with the AIMD algorithm.",
    "Write a Python program to parse and analyse Apache log files and report top IPs.",
    "Implement a graph traversal algorithm (BFS and DFS) in Python with cycle detection.",
    "Explain how attention mechanisms in LLMs scale and what KV cache does to reduce memory.",
    "Write a complete Terraform configuration to provision an EC2 instance with a VPC and security groups.",
    "Design a logging and observability stack for a microservices system and explain the tradeoffs.",
    "Write a Python implementation of a simple HTTP server from scratch using sockets.",
    "Explain how speculative decoding works in LLM inference and what speedups it achieves.",
    "Write a complete implementation of a trie data structure in Python with search and prefix matching.",
    "Design a job scheduling system for a distributed compute cluster and explain the algorithm.",
    "Write a Python script that connects to a PostgreSQL database and performs batch upserts efficiently.",
    "Explain the differences between OLTP and OLAP databases with examples of each.",
    "Write a Python class implementing a thread pool with a work queue and configurable workers.",
    "Explain how gradient descent works in neural networks with a worked numerical example.",
    "Write a complete implementation of a skip list in Python with insertion and search.",
    "Design a caching strategy for a high-traffic API and explain eviction policies.",
    "Write a Python asyncio implementation of a concurrent HTTP downloader with rate limiting.",
    "Explain the CAP theorem with a concrete example of each of the three tradeoff scenarios.",
    "Write a complete implementation of a min-heap in Python with heapify and extract-min.",
    "Design a database schema for an e-commerce platform and explain normalisation decisions.",
    "Write a Python program to implement a simple neural network from scratch using NumPy.",
    "Explain how OS scheduling algorithms work — FCFS, SJF, Round Robin — with examples.",
    "Write a complete gRPC service in Python for a payment processing system.",
    "Explain how columnar storage formats like Parquet improve query performance.",
    "Write a Python implementation of A* pathfinding with explanation of the heuristic.",
]

assert len(SHORT_PROMPTS) == 50, "Need exactly 50 short prompts"
assert len(LONG_PROMPTS)  == 50, "Need exactly 50 long prompts"


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    condition: str       # "fcfs" or "sjf"
    prompt_class: str    # "SHORT" or "LONG"
    prompt: str
    run_id: int
    latency_ms: float = 0.0
    status: int = 0
    error: str = ""


# ── Worker ────────────────────────────────────────────────────────────────────

def send(url: str, model: str, result: Result) -> None:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": result.prompt}],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result.status = resp.status
            resp.read()
    except urllib.error.HTTPError as e:
        result.status = e.code
        result.error = str(e)
    except Exception as e:
        result.error = str(e)
    finally:
        result.latency_ms = (time.time() - t0) * 1000


# ── Single run ────────────────────────────────────────────────────────────────

def run_condition(url: str, model: str, condition: str, run_id: int) -> list[Result]:
    """Fire all SHORT + LONG prompts concurrently, return results."""
    results: list[Result] = []
    for p in SHORT_PROMPTS:
        results.append(Result(condition=condition, prompt_class="SHORT",
                              prompt=p, run_id=run_id))
    for p in LONG_PROMPTS:
        results.append(Result(condition=condition, prompt_class="LONG",
                              prompt=p, run_id=run_id))

    threads = [
        threading.Thread(target=send, args=(url, model, r), daemon=True)
        for r in results
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: int) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def summarise(results: list[Result]) -> dict:
    by_condition: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.error:
            continue
        by_condition.setdefault(r.condition, {}).setdefault(r.prompt_class, []).append(r.latency_ms)

    rows = []
    for condition, classes in sorted(by_condition.items()):
        for prompt_class, latencies in sorted(classes.items()):
            rows.append({
                "condition":    condition,
                "prompt_class": prompt_class,
                "n":            len(latencies),
                "p50_ms":       round(percentile(latencies, 50), 1),
                "p95_ms":       round(percentile(latencies, 95), 1),
                "p99_ms":       round(percentile(latencies, 99), 1),
                "mean_ms":      round(statistics.mean(latencies), 1),
            })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clairvoyant FCFS vs SJF benchmark")
    parser.add_argument("--model",     default="gemma3:4b")
    parser.add_argument("--runs",      type=int, default=5)
    parser.add_argument("--condition", choices=["fcfs", "sjf", "both"], default="both")
    parser.add_argument("--out",       default="benchmark_results.csv")
    parser.add_argument("--summary",   default="benchmark_summary.csv")
    parser.add_argument("--fcfs-url",  default=FCFS_URL)
    parser.add_argument("--sjf-url",   default=SJF_URL)
    args = parser.parse_args()

    conditions = []
    if args.condition in ("fcfs", "both"):
        conditions.append(("fcfs", args.fcfs_url))
    if args.condition in ("sjf", "both"):
        conditions.append(("sjf", args.sjf_url))

    all_results: list[Result] = []

    for condition, url in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition.upper()}  url={url}  model={args.model}")
        print(f"{'='*60}")
        for run_id in range(1, args.runs + 1):
            print(f"\n  Run {run_id}/{args.runs} — firing 100 concurrent requests...", flush=True)
            t0 = time.time()
            results = run_condition(url, args.model, condition, run_id)
            elapsed = time.time() - t0

            ok    = [r for r in results if not r.error]
            errs  = [r for r in results if r.error]
            short = [r for r in ok if r.prompt_class == "SHORT"]
            long_ = [r for r in ok if r.prompt_class == "LONG"]

            print(f"  Done in {elapsed:.1f}s — {len(ok)} ok, {len(errs)} errors")
            if short:
                print(f"  SHORT  p50={percentile([r.latency_ms for r in short], 50)/1000:.2f}s  "
                      f"p95={percentile([r.latency_ms for r in short], 95)/1000:.2f}s  "
                      f"p99={percentile([r.latency_ms for r in short], 99)/1000:.2f}s")
            if long_:
                print(f"  LONG   p50={percentile([r.latency_ms for r in long_], 50)/1000:.2f}s  "
                      f"p95={percentile([r.latency_ms for r in long_], 95)/1000:.2f}s")
            if errs:
                print(f"  ERRORS: {errs[0].error[:80]}", file=sys.stderr)

            all_results.extend(results)

    # Write raw results
    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "prompt_class", "run_id", "latency_ms", "status", "error", "prompt"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "condition":    r.condition,
                "prompt_class": r.prompt_class,
                "run_id":       r.run_id,
                "latency_ms":   round(r.latency_ms, 2),
                "status":       r.status,
                "error":        r.error,
                "prompt":       r.prompt[:80],
            })
    print(f"\nRaw results → {out_path}")

    # Write summary
    summary_rows = summarise(all_results)
    summary_path = Path(args.summary)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "prompt_class", "n", "p50_ms", "p95_ms", "p99_ms", "mean_ms"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary      → {summary_path}\n")
    print(f"{'CONDITION':<8} {'CLASS':<8} {'N':>5} {'P50(s)':>8} {'P95(s)':>8} {'P99(s)':>8}")
    print("─" * 55)
    for row in summary_rows:
        print(f"{row['condition']:<8} {row['prompt_class']:<8} {row['n']:>5} "
              f"{row['p50_ms']/1000:>8.2f} {row['p95_ms']/1000:>8.2f} {row['p99_ms']/1000:>8.2f}")


if __name__ == "__main__":
    main()
