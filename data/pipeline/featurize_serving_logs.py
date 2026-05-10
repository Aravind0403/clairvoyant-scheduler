"""
featurize_serving_logs.py — Convert serving_logs.jsonl to model_d_training_data.csv

Applies the exact same 19-feature extraction as featurize.py.
Labels by actual Gemma output token count (same thresholds).
Balances classes to min(n_per_class, --cap).
Output is drop-in compatible with train.py.

Usage:
    python data/pipeline/featurize_serving_logs.py \\
        [--in  data/serving_logs.jsonl] \\
        [--out data/model_d_training_data.csv] \\
        [--cap 500]
"""

import argparse
import csv
import json
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

# Use len//4 at runtime (same as Go scheduler) — no tokenizer dependency
def prompt_token_len_approx(text: str) -> int:
    return len(text) // 4

def extract_features(prompt: str, actual_output_tokens: int) -> dict:
    words = set(re.findall(r"\b\w+\b", prompt.lower()))

    instruction_verb = "other"
    for label, pattern in VERB_PATTERNS:
        if pattern.search(prompt):
            instruction_verb = label
            break

    prompt_clean = prompt.rstrip()
    ends_with_question = 1 if prompt_clean and prompt_clean.endswith("?") else 0

    clause_count = (
        prompt.count(',')
        + prompt.count(';')
        + prompt.count(' and ')
        + prompt.count(' but ')
        + prompt.count(' because ')
    )

    has_format = int(any(kw in prompt.lower() for kw in FORMAT_KEYWORDS))

    return {
        "prompt":                prompt,
        "prompt_token_len":      prompt_token_len_approx(prompt),
        "has_code_keyword":      int(bool(words & CODE_KEYWORDS)),
        "has_length_constraint": int(bool(CONSTRAINT_PATTERNS.search(prompt))),
        "instruction_verb":      instruction_verb,
        "ends_with_question":    ends_with_question,
        "has_format_keyword":    has_format,
        "clause_count":          clause_count,
        "actual_output_tokens":  actual_output_tokens,
    }


# ── Label thresholds — same as train.py ──────────────────────────────────────
def label(token_count: int) -> int:
    if token_count < 200:
        return 0   # Short
    elif token_count < 800:
        return 1   # Medium
    return 2       # Long


LABEL_NAMES = {0: "Short", 1: "Medium", 2: "Long"}

FIELDNAMES = [
    "prompt", "prompt_token_len", "has_code_keyword",
    "has_length_constraint", "instruction_verb", "ends_with_question",
    "has_format_keyword", "clause_count", "actual_output_tokens",
]


# ── Main ─────────────────────────────────────────────────────────────────────
def main(in_path: str, out_path: str, cap: int):
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run run_serving_logs.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {in_path}...")
    buckets = {0: [], 1: [], 2: []}
    skipped = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt    = rec.get("prompt", "").strip()
            tok_count = rec.get("response_token_count", 0)

            if not prompt or tok_count == 0:
                skipped += 1
                continue

            feats = extract_features(prompt, tok_count)
            lbl   = label(tok_count)
            buckets[lbl].append(feats)

    for lbl, name in LABEL_NAMES.items():
        print(f"  {name}: {len(buckets[lbl]):,}")
    print(f"  Skipped: {skipped}")

    # balance
    random.seed(42)
    min_class = min(len(b) for b in buckets.values())
    n_per_class = min(min_class, cap)

    if n_per_class == 0:
        print("ERROR: one or more classes is empty. Collect more prompts.", file=sys.stderr)
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

    print(f"\nSaved {len(rows):,} rows ({n_per_class} per class) → {out_path}")
    print(f"Ready for: python model/train.py --data {out_path} "
          f"--out model/predictor_model_d.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize serving logs for Model D")
    parser.add_argument("--in",  dest="in_path",  default="data/serving_logs.jsonl")
    parser.add_argument("--out", dest="out_path",  default="data/model_d_training_data.csv")
    parser.add_argument("--cap", type=int,         default=500,
                        help="Max rows per class (default: 500)")
    args = parser.parse_args()
    main(args.in_path, args.out_path, args.cap)
