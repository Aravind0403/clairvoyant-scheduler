"""
collect_code_alpaca.py — Build training data from CodeAlpaca-20K

Covers the code-generation blind spot absent in ShareGPT/LMSYS/Alpaca.
No login required. Downloads in seconds.
Uses existing output text — no Ollama needed.

Dataset: sahil2801/CodeAlpaca-20k
  - 20K code-focused instruction-output pairs
  - All instructions are code-related (write function, debug, explain algorithm, etc.)
  - has_code_keyword will dominate — intentional, tests code workload generalisation

Expected class distribution (code outputs tend to be medium-length):
  Short  (< 200 tokens)  : lower proportion
  Medium (200–799 tokens) : dominant
  Long   (≥ 800 tokens)   : moderate

Usage:
    python data/pipeline/collect_code_alpaca.py \\
        [--cap 2000] \\
        [--out data/code_alpaca_labeled.csv]

Then retrain:
    python model/train.py \\
        --data data/code_alpaca_labeled.csv \\
        --out  model/predictor_code_alpaca.json

For the cross-distribution matrix:
  - Use as both a training row AND a test column
  - Off-diagonal entries reveal: does training on chat data generalise to code workloads?
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

def main(cap: int, out_path: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print("Downloading CodeAlpaca-20K (no login required)...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    print(f"  ✓ Loaded {len(ds):,} rows")

    buckets = {0: [], 1: [], 2: []}
    skipped = 0

    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp         = (row.get("input") or "").strip()
        output      = (row.get("output") or "").strip()

        prompt = instruction
        if inp:
            prompt = f"{instruction}\n\nInput:\n{inp}"

        if not prompt or not output:
            skipped += 1
            continue

        if not (10 <= len(prompt) <= 2000):
            skipped += 1
            continue

        resp_tokens = len(output) // 4
        if resp_tokens == 0:
            skipped += 1
            continue

        if resp_tokens < 200:
            lbl = 0
        elif resp_tokens < 800:
            lbl = 1
        else:
            lbl = 2

        feats = extract_features(prompt, resp_tokens)
        buckets[lbl].append(feats)

    print(f"\nRaw counts — Short={len(buckets[0])}, "
          f"Medium={len(buckets[1])}, Long={len(buckets[2])}, "
          f"Skipped={skipped}")

    # balance
    random.seed(42)
    min_class   = min(len(b) for b in buckets.values())
    n_per_class = min(min_class, cap)

    if n_per_class == 0:
        print("\nERROR: one or more classes is empty.", file=sys.stderr)
        print("Possible cause: code outputs cluster in Medium. Adjust thresholds?", file=sys.stderr)
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
    print(f"\nExpected insight: CodeAlpaca-trained model should rank code prompts "
          f"accurately. Cross-test on ShareGPT/LMSYS reveals generalisation gap.")
    print(f"\nNext — retrain:")
    print(f"  python model/train.py --data {out_path} --out model/predictor_code_alpaca.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build training data from CodeAlpaca-20K (no login needed)"
    )
    parser.add_argument("--cap", type=int, default=2000,
                        help="Max rows per class (default: 2000)")
    parser.add_argument("--out", default="data/code_alpaca_labeled.csv")
    args = parser.parse_args()
    main(args.cap, args.out)
