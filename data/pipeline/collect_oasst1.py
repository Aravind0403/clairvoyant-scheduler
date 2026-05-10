"""
collect_oasst1.py — Build training data from OpenAssistant OASST1

No login required. Downloads in seconds.
Covers human-annotated instruction responses — more naturally varied in length
than GPT-generated datasets (Alpaca, CodeAlpaca, Dolly) which are Short-biased.

Dataset: OpenAssistant/oasst1 (84K messages, tree structure)
  Long distribution check:
    <200 tokens  : 34,912  (66%)
    200-799 tokens: 17,449  (33%)
    >=800 tokens :    551   (1%)
  → 551 Long examples — viable as a balanced training source (cap ≤ 500).

Structure: each row is an individual message with parent_id linking to its parent.
We pair each English assistant turn with its parent human turn as the prompt.
Filters to English-only, single-turn (first human → first assistant) for consistency
with ShareGPT and LMSYS collection strategies.

Usage:
    python data/pipeline/collect_oasst1.py \\
        [--cap 500] \\
        [--out data/oasst1_labeled.csv]

Then retrain:
    python model/train.py \\
        --data data/oasst1_labeled.csv \\
        --out  model/predictor_oasst1.json
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

    print("Downloading OpenAssistant OASST1 (no login required)...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    print(f"  ✓ Loaded {len(ds):,} messages")

    # Build message_id → row lookup for parent resolution
    print("  Building parent lookup...")
    id_to_row = {row["message_id"]: row for row in ds}

    buckets = {0: [], 1: [], 2: []}
    skipped = {"lang": 0, "no_parent": 0, "not_human": 0, "length": 0, "empty": 0}

    for row in ds:
        # We want assistant turns only
        if row.get("role") != "assistant":
            continue

        # English only
        if row.get("lang", "en") != "en":
            skipped["lang"] += 1
            continue

        response = (row.get("text") or "").strip()
        if not response:
            skipped["empty"] += 1
            continue

        # Look up parent (the human turn)
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in id_to_row:
            skipped["no_parent"] += 1
            continue

        parent_row = id_to_row[parent_id]
        if parent_row.get("role") != "prompter":
            # Parent is another assistant turn (multi-assistant thread) — skip
            skipped["not_human"] += 1
            continue

        prompt = (parent_row.get("text") or "").strip()
        if not prompt:
            skipped["empty"] += 1
            continue

        if not (10 <= len(prompt) <= 2000):
            skipped["length"] += 1
            continue

        resp_tokens = len(response) // 4
        if resp_tokens == 0:
            skipped["empty"] += 1
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
          f"Medium={len(buckets[1])}, Long={len(buckets[2])}")
    print(f"Skipped: {skipped}")

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

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total:,} rows ({n_per_class} per class) → {out_path}")
    print(f"\nNote: Long class is capped at {min(len(buckets[2]), cap)} "
          f"(only {len(buckets[2])} Long turns in OASST1 English train split).")
    print(f"\nNext — retrain:")
    print(f"  python model/train.py --data {out_path} --out model/predictor_oasst1.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build training data from OpenAssistant OASST1 (no login needed)"
    )
    parser.add_argument("--cap", type=int, default=500,
                        help="Max rows per class (default: 500, Long class has ~500 total)")
    parser.add_argument("--out", default="data/oasst1_labeled.csv")
    args = parser.parse_args()
    main(args.cap, args.out)
