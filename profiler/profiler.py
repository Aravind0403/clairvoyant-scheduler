"""Main profiling loop for the Clairvoyant Scheduler.

Runs every prompt in prompts.ALL through Ollama (gemma3:4b by default),
extracts features, and appends results to training_data.csv.

Usage
-----
    python profiler.py [--model gemma3:4b] [--out training_data.csv] [--runs N]

--runs N  repeats the full prompt list N times to collect more samples
          with natural latency variance (default: 1).
"""

import argparse
import csv
import pathlib
import sys

from llm_client import generate, health_check, LLMResponse, DEFAULT_MODEL
from feature_extractor import extract, approx_tokens, Features
from prompts import ALL as ALL_PROMPTS


CSV_FIELDS = [
    "prompt",
    "prompt_token_len",
    "has_code_keyword",
    "has_length_constraint",
    "instruction_verb",
    "ends_with_question",
    "actual_output_tokens",
    "latency_ms",
]


def _open_writer(path: pathlib.Path) -> tuple[csv.DictWriter, object]:
    exists = path.exists() and path.stat().st_size > 0
    fh = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
    if not exists:
        writer.writeheader()
    return writer, fh


def run(prompts: list[str], model: str, out_path: pathlib.Path, runs: int) -> None:
    writer, fh = _open_writer(out_path)
    total = len(prompts) * runs
    completed = 0

    try:
        for run_num in range(1, runs + 1):
            for prompt in prompts:
                completed += 1
                preview = prompt[:55].replace("\n", " ")
                print(f"[{completed}/{total}] run={run_num} › {preview!r}…", flush=True)

                try:
                    resp = generate(prompt, model=model)
                except Exception as exc:
                    print(f"  ✗ ERROR: {exc}", file=sys.stderr)
                    continue

                feats = extract(prompt)
                output_tokens = approx_tokens(resp.text)

                row = {
                    "prompt": prompt,
                    **feats.to_dict(),
                    "actual_output_tokens": output_tokens,
                    "latency_ms": resp.latency_ms,
                }
                writer.writerow(row)
                fh.flush()

                print(
                    f"  ✓ tokens_in={feats.prompt_token_len:>3}  "
                    f"tokens_out={output_tokens:>4}  "
                    f"latency={resp.latency_ms:>8.1f} ms  "
                    f"verb={feats.instruction_verb}"
                )
    finally:
        fh.close()

    print(f"\nDone — {completed} rows written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clairvoyant Scheduler — LLM profiler")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--out",   default="training_data.csv", help="Output CSV path")
    parser.add_argument("--runs",  type=int, default=1,
                        help="Number of full passes over the prompt list")
    args = parser.parse_args()

    health_check(model=args.model)

    run(
        prompts=ALL_PROMPTS,
        model=args.model,
        out_path=pathlib.Path(args.out),
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
