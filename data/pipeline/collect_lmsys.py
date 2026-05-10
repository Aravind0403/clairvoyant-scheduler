"""
collect_lmsys.py — Sample prompts from LMSYS-Chat-1M for Model D

Downloads via HuggingFace streaming (no full 1M-row download needed).
Filters to single-turn, English, prompt length 10–2000 chars.
Saves sampled prompts to data/lmsys_prompts.jsonl.

Usage:
    python data/pipeline/collect_lmsys.py [--n 2000] [--out data/lmsys_prompts.jsonl]
"""

import argparse
import json
import os
import re
import sys

def main(n_target: int, out_path: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets --break-system-packages", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    print(f"Streaming LMSYS-Chat-1M (target: {n_target} prompts)...")
    print("This may take a few minutes on first run (index download).")

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    scanned = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1

            if scanned % 10000 == 0:
                print(f"  scanned {scanned:,} — kept {saved:,}")

            if saved >= n_target:
                break

            # ── filters ──────────────────────────────────────────────────────
            # single-turn only (one human + one assistant message)
            turns = row.get("conversation", [])
            if len(turns) != 2:
                continue
            if turns[0].get("role") != "human" or turns[1].get("role") != "assistant":
                continue

            # English only
            if row.get("language", "English") != "English":
                continue

            prompt = turns[0].get("content", "").strip()
            if not prompt:
                continue

            # length filter — matches clean.py criteria
            if not (10 <= len(prompt) <= 2000):
                continue

            # skip prompts that are just code blocks — no lexical signal
            if prompt.startswith("```"):
                continue

            record = {
                "prompt": prompt,
                "source": "lmsys-chat-1m",
                "conversation_id": row.get("conversation_id", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\nDone. Scanned {scanned:,} rows, saved {saved:,} prompts → {out_path}")
    if saved < n_target:
        print(f"WARNING: only got {saved} prompts (wanted {n_target}). "
              f"Try increasing --n or relaxing filters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample LMSYS-Chat-1M prompts")
    parser.add_argument("--n",   type=int, default=2000,
                        help="Number of prompts to collect (default: 2000)")
    parser.add_argument("--out", default="data/lmsys_prompts.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()
    main(args.n, args.out)
