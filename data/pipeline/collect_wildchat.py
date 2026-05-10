"""
collect_wildchat.py — Build in-the-wild test data from WildChat-1M

In-the-wild real user conversations with GPT-4.
REQUIRES HuggingFace login (gated dataset).

Dataset: allenai/WildChat-1M
  - ~1M real-world conversations from diverse users
  - More adversarial, ambiguous, and multi-lingual than curated datasets
  - Excellent for testing generalisation to real production traffic

Access:
  1. Visit https://huggingface.co/datasets/allenai/WildChat-1M
     and accept the dataset terms (requires HF account)
  2. Run: huggingface-cli login
  3. Re-run this script

Filters applied:
  - English only
  - First human turn + first assistant turn only (ignore multi-turn)
  - Prompt length 10–2000 chars
  - Response token count > 0

Expected class distribution:
  - GPT-4 responses tend to be longer → more Medium/Long than chat datasets
  - This makes it a good stress-test for Short-class precision

Use as TEST-ONLY column — GPT-4 length distribution ≠ Gemma distribution.

Usage:
    python data/pipeline/collect_wildchat.py \\
        [--n 50000] \\
        [--cap 500] \\
        [--out data/wildchat_test.csv]

Then evaluate:
    python model/evaluate_ranking.py \\
        --model model/predictor_v3.json \\
        --test  data/wildchat_test.csv
"""

import argparse
import csv
import os
import random
import re
import sys

# ── Feature extraction — mirrors featurize.py exactly ────────────────────────

FORMAT_KEYWORDS = [
    "list", "table", "bullet", "step by step", "enumerate",
    "outline", "numbered", "format", "structure",
]

CODE_KEYWORDS = frozenset([
    "code", "function", "def", "class", "script", "program",
    "implement", "algorithm", "debug", "refactor", "import",
    "variable", "loop", "recursion", "api", "sql", "regex",
])

CONSTRAINT_PATTERNS = re.compile(
    r"\bin\s+\d+\s+words?\b"
    r"|\bin\s+\d+\s+sentences?\b"
    r"|\bno\s+more\s+than\s+\d+"
    r"|\bunder\s+\d+\s+words?\b"
    r"|\bbriefly\b"
    r"|\bconcisely\b"
    r"|\bshortly\b"
    r"|\bone[\s-]liner\b"
    r"|\btl;?dr\b",
    re.I,
)

VERB_PATTERNS = [
    ("summarize",  re.compile(r"\bsummariz(?:e|ing)\b",    re.I)),
    ("explain",    re.compile(r"\bexplain\b",               re.I)),
    ("compare",    re.compile(r"\bcompar(?:e|ing)\b",       re.I)),
    ("translate",  re.compile(r"\btranslat(?:e|ing)\b",     re.I)),
    ("generate",   re.compile(r"\bgenerat(?:e|ing)\b",      re.I)),
    ("implement",  re.compile(r"\bimplement\b",             re.I)),
    ("debug",      re.compile(r"\bdebug\b",                 re.I)),
    ("refactor",   re.compile(r"\brefactor\b",              re.I)),
    ("list",       re.compile(r"\blist\b",                  re.I)),
    ("write",      re.compile(r"\bwrite\b",                 re.I)),
    ("describe",   re.compile(r"\bdescrib(?:e|ing)\b",      re.I)),
    ("define",     re.compile(r"\bdefin(?:e|ing)\b",        re.I)),
    ("what",       re.compile(r"\bwhat\b",                  re.I)),
    ("how",        re.compile(r"\bhow\b",                   re.I)),
    ("why",        re.compile(r"\bwhy\b",                   re.I)),
]


def extract_features(prompt: str, actual_output_tokens: int) -> dict:
    words = set(re.findall(r"\b\w+\b", prompt.lower()))

    instruction_verb = "other"
    for label, pattern in VERB_PATTERNS:
        if pattern.search(prompt):
            instruction_verb = label
            break

    ends_with_question = 1 if prompt.rstrip().endswith("?") else 0
    clause_count = (
        prompt.count(',') + prompt.count(';')
        + prompt.count(' and ') + prompt.count(' but ')
        + prompt.count(' because ')
    )

    return {
        "prompt":                prompt,
        "prompt_token_len":      len(prompt) // 4,
        "has_code_keyword":      int(bool(words & CODE_KEYWORDS)),
        "has_length_constraint": int(bool(CONSTRAINT_PATTERNS.search(prompt))),
        "instruction_verb":      instruction_verb,
        "ends_with_question":    ends_with_question,
        "has_format_keyword":    int(any(kw in prompt.lower() for kw in FORMAT_KEYWORDS)),
        "clause_count":          clause_count,
        "actual_output_tokens":  actual_output_tokens,
    }


FIELDNAMES = [
    "prompt", "prompt_token_len", "has_code_keyword",
    "has_length_constraint", "instruction_verb", "ends_with_question",
    "has_format_keyword", "clause_count", "actual_output_tokens",
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main(n_scan: int, cap: int, out_path: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming WildChat-1M (scanning up to {n_scan:,} rows)...")

    try:
        ds = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
        )
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("gated", "not found", "access", "authentication")):
            print("\nERROR: WildChat-1M is a gated dataset.", file=sys.stderr)
            print("Fix:", file=sys.stderr)
            print("  1. Visit https://huggingface.co/datasets/allenai/WildChat-1M", file=sys.stderr)
            print("     and accept the dataset terms (requires HF account)", file=sys.stderr)
            print("  2. Run: huggingface-cli login", file=sys.stderr)
            print("  3. Re-run this script", file=sys.stderr)
        else:
            print(f"\nERROR loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    buckets  = {0: [], 1: [], 2: []}
    scanned  = 0
    kept     = 0
    filtered = {"lang": 0, "turns": 0, "length": 0, "empty": 0}

    for row in ds:
        if scanned >= n_scan:
            break
        if all(len(b) >= cap for b in buckets.values()):
            print(f"  All classes at cap ({cap}) — stopping early at {scanned:,} scanned")
            break

        scanned += 1
        if scanned % 5000 == 0:
            counts = {k: len(v) for k, v in buckets.items()}
            print(f"  scanned {scanned:,} | kept {kept:,} | "
                  f"S={counts[0]} M={counts[1]} L={counts[2]}")

        # English only
        if row.get("language", "English") != "English":
            filtered["lang"] += 1
            continue

        conversation = row.get("conversation", [])
        if len(conversation) < 2:
            filtered["turns"] += 1
            continue

        # First human + first assistant turn
        human_turn     = next((t for t in conversation if t.get("role") == "user"), None)
        assistant_turn = next((t for t in conversation if t.get("role") == "assistant"), None)

        if not human_turn or not assistant_turn:
            filtered["turns"] += 1
            continue

        prompt   = (human_turn.get("content") or "").strip()
        response = (assistant_turn.get("content") or "").strip()

        if not prompt or not response:
            filtered["empty"] += 1
            continue

        if not (10 <= len(prompt) <= 2000):
            filtered["length"] += 1
            continue

        resp_tokens = len(response) // 4
        if resp_tokens == 0:
            filtered["empty"] += 1
            continue

        if resp_tokens < 200:
            lbl = 0
        elif resp_tokens < 800:
            lbl = 1
        else:
            lbl = 2

        if len(buckets[lbl]) >= cap:
            continue

        feats = extract_features(prompt, resp_tokens)
        buckets[lbl].append(feats)
        kept += 1

    print(f"\nScanned: {scanned:,}  Kept: {kept:,}")
    print(f"Filter breakdown: {filtered}")
    for lbl, name in {0: "Short", 1: "Medium", 2: "Long"}.items():
        print(f"  {name}: {len(buckets[lbl]):,}")

    random.seed(42)
    min_class   = min(len(b) for b in buckets.values())
    n_per_class = min(min_class, cap)

    if n_per_class == 0:
        print("\nERROR: one or more classes is empty.", file=sys.stderr)
        print("Try increasing --n or check that you've accepted the dataset terms.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for lbl in [0, 1, 2]:
        random.shuffle(buckets[lbl])
        rows.extend(buckets[lbl][:n_per_class])
    random.shuffle(rows)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total:,} rows ({n_per_class} per class) → {out_path}")
    print(f"\nNote: GPT-4 response lengths ≠ Gemma response lengths.")
    print(f"Use as test-only cross-distribution column, not a training source.")
    print(f"\nEvaluate with:")
    print(f"  python model/evaluate_ranking.py --model model/predictor_v3.json "
          f"--test {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build in-the-wild test data from WildChat-1M (requires HF login)"
    )
    parser.add_argument("--n",   type=int, default=50_000,
                        help="Max rows to scan (default: 50000)")
    parser.add_argument("--cap", type=int, default=500,
                        help="Max rows per class (default: 500)")
    parser.add_argument("--out", default="data/wildchat_test.csv")
    args = parser.parse_args()
    main(args.n, args.cap, args.out)
