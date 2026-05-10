"""
collect_dolly.py — Build training data from Databricks Dolly 15K

No login, no gating, downloads in seconds.
Uses existing response text — no Ollama needed.
Covers 8 instruction categories: brainstorming, classification, closed_qa,
generation, information_extraction, open_qa, summarization, creative_writing.

Produces a balanced CSV compatible with train.py.

Runtime: ~2-3 minutes on M1 (tokeniser warm-up + 15K rows)

Usage:
    python data/pipeline/collect_dolly.py \\
        [--cap 500] \\
        [--out data/dolly_labeled.csv]

Then retrain:
    python model/train.py \\
        --data data/dolly_labeled.csv \\
        --out  model/predictor_model_dolly.json

Or just use for ordering test — swap ONNX in scheduler and run
test_queue_ordering.py with Dolly-sourced SHORT/LONG prompts.
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

    print("Downloading Databricks Dolly 15K (no login required)...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"  ✓ Loaded {len(ds):,} rows")

    # show category distribution
    from collections import Counter
    cats = Counter(r["category"] for r in ds)
    print("  Categories:", dict(cats))

    buckets  = {0: [], 1: [], 2: []}
    skipped  = 0
    by_cat   = {}   # track which categories produce which classes

    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("context") or "").strip()
        response    = (row.get("response") or "").strip()
        category    = row.get("category", "unknown")

        # combine instruction + context as the prompt (matches real usage)
        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContext: {context}"

        if not prompt or not response:
            skipped += 1
            continue

        if not (10 <= len(prompt) <= 2000):
            skipped += 1
            continue

        # len//4 for response — no truncation, same as Go scheduler
        resp_tokens = len(response) // 4
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
        feats["category"] = category   # keep for analysis, strip before CSV

        buckets[lbl].append(feats)
        by_cat.setdefault(category, {0: 0, 1: 0, 2: 0})
        by_cat[category][lbl] += 1

    label_names = {0: "Short", 1: "Medium", 2: "Long"}
    print(f"\nRaw counts — Short={len(buckets[0])}, "
          f"Medium={len(buckets[1])}, Long={len(buckets[2])}, "
          f"Skipped={skipped}")

    print("\nClass distribution by category:")
    for cat, counts in sorted(by_cat.items()):
        print(f"  {cat:<25} S={counts[0]:>4}  M={counts[1]:>4}  L={counts[2]:>4}")

    # balance
    random.seed(42)
    min_class   = min(len(b) for b in buckets.values())
    n_per_class = min(min_class, cap)

    if n_per_class == 0:
        print("\nERROR: one or more classes is empty.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for lbl in [0, 1, 2]:
        random.shuffle(buckets[lbl])
        rows.extend(buckets[lbl][:n_per_class])
    random.shuffle(rows)

    # strip category before writing (not a model feature)
    for r in rows:
        r.pop("category", None)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total:,} rows ({n_per_class} per class) → {out_path}")
    print(f"\nNext — retrain:")
    print(f"  python model/train.py --data {out_path} "
          f"--out model/predictor_dolly.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build training data from Dolly 15K (no login needed)"
    )
    parser.add_argument("--cap", type=int, default=500,
                        help="Max rows per class (default: 500, Dolly is only 15K total)")
    parser.add_argument("--out", default="data/dolly_labeled.csv")
    args = parser.parse_args()
    main(args.cap, args.out)
