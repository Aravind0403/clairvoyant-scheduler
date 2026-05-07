"""
Queue ordering integration test for the Clairvoyant Scheduler.

Fires SHORT and LONG requests concurrently and verifies that Short requests
finish before Long ones — the core SJF guarantee.

Usage:
    python test_queue_ordering.py [--url http://localhost:8080] [--model gemma3:4b]
"""

import argparse
import json
import statistics
import threading
import time
from dataclasses import dataclass, field

import urllib.request
import urllib.error

# ── Prompts ───────────────────────────────────────────────────────────────────

SHORT_PROMPTS = [
    "What's the best programming language for metamorphic programming",
    "Brainstorm ideas for work from home jobs for seniors",
    "What arrays do not contain duplicate elements in Java?",
    "what do you know about software reverse engineering?",
]

LONG_PROMPTS = [
    "can you help me make a tic tac toe game using react?",
    "can you please write me a program in python that can read from an email and load an attached xml file into a postgresql database",
    "give me the C++ ROS source code to move a servo in a Robotis OP3 robot",
    "Create a Tooltip in NextJS using React Context to close the previously opened Tooltip when a new one opens",
]

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Result:
    label: str        # "SHORT" or "LONG"
    prompt: str
    start: float      # time.time() when request was fired
    end: float = 0.0  # time.time() when response arrived
    status: int = 0
    error: str = ""
    predicted_class: int = -1  # parsed from scheduler log (not available here)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def finish_offset(self) -> float:
        """Seconds after the batch start this request completed."""
        return self.end  # will be normalised against batch_start after collection


# ── Worker ────────────────────────────────────────────────────────────────────

def send_request(url: str, model: str, result: Result) -> None:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": result.prompt}],
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result.start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result.status = resp.status
            result.end = time.time()
    except urllib.error.HTTPError as e:
        result.status = e.code
        result.end = time.time()
        result.error = str(e)
    except Exception as e:
        result.end = time.time()
        result.error = str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default="http://[::1]:8080/v1/chat/completions")
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    results: list[Result] = []
    for p in SHORT_PROMPTS:
        results.append(Result(label="SHORT", prompt=p, start=0.0))
    for p in LONG_PROMPTS:
        results.append(Result(label="LONG",  prompt=p, start=0.0))

    print(f"Firing {len(SHORT_PROMPTS)} SHORT + {len(LONG_PROMPTS)} LONG requests concurrently")
    print(f"Target: {args.url}  model: {args.model}\n")

    threads = [
        threading.Thread(target=send_request, args=(args.url, args.model, r), daemon=True)
        for r in results
    ]

    batch_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    batch_end = time.time()

    # Normalise finish times to seconds after batch start
    for r in results:
        r.end = r.end - batch_start
        r.start = r.start - batch_start

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"{'LABEL':<8} {'FINISH(s)':>10} {'DURATION(s)':>12}  PROMPT (first 60 chars)")
    print("─" * 80)

    sorted_results = sorted(results, key=lambda r: r.end)
    for r in sorted_results:
        err = f"  ← ERROR {r.error}" if r.error else ""
        print(f"{r.label:<8} {r.end:>10.2f} {r.duration:>12.2f}  {r.prompt[:60]!r}{err}")

    # ── Summary ───────────────────────────────────────────────────────────────
    short_ok = [r for r in results if r.label == "SHORT" and not r.error]
    long_ok  = [r for r in results if r.label == "LONG"  and not r.error]

    if not short_ok or not long_ok:
        print("\nNot enough successful results to summarise.")
        return

    short_finish = [r.end for r in short_ok]
    long_finish  = [r.end for r in long_ok]

    short_avg = statistics.mean(short_finish)
    long_avg  = statistics.mean(long_finish)

    short_finished_first = sum(
        1 for s in short_finish for l in long_finish if s < l
    )
    total_pairs = len(short_finish) * len(long_finish)
    ordering_accuracy = short_finished_first / total_pairs * 100

    print("\n" + "─" * 80)
    print("SUMMARY")
    print(f"  Short avg finish : {short_avg:.2f}s")
    print(f"  Long  avg finish : {long_avg:.2f}s")
    print(f"  Advantage        : {long_avg - short_avg:+.2f}s")
    print(f"  Ordering accuracy: {short_finished_first}/{total_pairs} pairs = {ordering_accuracy:.1f}%")
    print(f"  Total wall time  : {batch_end - batch_start:.2f}s")
    print()

    if ordering_accuracy >= 80:
        print("✓ PASS — Short requests finished before Long in the majority of pairs")
    elif ordering_accuracy >= 50:
        print("~ PARTIAL — Some ordering benefit but not consistent")
    else:
        print("✗ FAIL — Long requests finishing before Short (check predictor / queue)")


if __name__ == "__main__":
    main()
