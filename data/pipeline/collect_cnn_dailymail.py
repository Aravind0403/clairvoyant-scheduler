"""
collect_cnn_dailymail.py — Build RAG/Summarisation test data from CNN/DailyMail

No login required. Covers the summarisation blind spot.

Dataset: cnn_dailymail (version 3.0.0)
  - ~300K news article + highlight pairs
  - Prompt  = "Summarise the following article:\n\n[article_truncated]"
  - Response = highlights (the gold summary)

Expected class distribution:
  Short  (< 200 tokens)  : dominant — highlights are concise by design
  Medium (200–799 tokens) : some
  Long   (≥ 800 tokens)   : rare

IMPORTANT: Due to expected class imbalance (mostly Short), this dataset is
recommended as a TEST-ONLY column in the cross-distribution matrix.
Do NOT train on it — the Short-heavy distribution will bias the classifier.

Use it to answer: does a model trained on chat data still rank RAG prompts correctly?
A good scheduler should predict Short for summarisation tasks even without seeing
RAG workloads during training.

Usage:
    python data/pipeline/collect_cnn_dailymail.py \\
        [--cap 500] \\
        [--out data/cnn_dailymail_test.csv]

Then evaluate:
    python model/evaluate_ranking.py \\
        --model model/predictor_v3.json \\
        --test  data/cnn_dailymail_test.csv
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

# Max article chars to include in prompt (keeps prompts under 2000-char filter)
ARTICLE_TRUNCATE = 1500
PROMPT_TEMPLATE  = "Summarise the following article:\n\n{article}"


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

    print("Downloading CNN/DailyMail 3.0.0 (no login required)...")
    # Use train split — we only need a test sample, train split is largest
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    print(f"  ✓ Loaded {len(ds):,} rows from test split")

    buckets = {0: [], 1: [], 2: []}
    skipped = 0

    for row in ds:
        article    = (row.get("article") or "").strip()
        highlights = (row.get("highlights") or "").strip()

        if not article or not highlights:
            skipped += 1
            continue

        # Truncate article to keep prompt within 2000-char filter
        article_trunc = article[:ARTICLE_TRUNCATE]
        prompt = PROMPT_TEMPLATE.format(article=article_trunc)

        if not (10 <= len(prompt) <= 2000):
            skipped += 1
            continue

        resp_tokens = len(highlights) // 4
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

    if len(buckets[0]) == 0 and len(buckets[1]) == 0 and len(buckets[2]) == 0:
        print("\nERROR: no rows collected.", file=sys.stderr)
        sys.exit(1)

    # For test-only use: do NOT balance — preserve natural distribution
    # This reveals the true class mix of RAG workloads
    print("\nNOTE: Not balancing classes — preserving natural RAG distribution for test-only use.")
    print("This is intentional: CNN/DailyMail is a TEST column, not a training source.")

    # But cap each class to avoid one class overwhelming ranking evaluation
    random.seed(42)
    rows = []
    for lbl in [0, 1, 2]:
        random.shuffle(buckets[lbl])
        rows.extend(buckets[lbl][:cap])
    random.shuffle(rows)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total:,} rows → {out_path}")
    print(f"\nNatural distribution shows what a RAG workload looks like:")
    print(f"  Short  (expected dominant): highlights are brief")
    print(f"  Medium : longer multi-paragraph summaries")
    print(f"  Long   : rare — highlights rarely exceed 800 tokens")
    print(f"\nUse as test column only:")
    print(f"  python model/evaluate_ranking.py --model model/predictor_v3.json "
          f"--test {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAG/Summarisation test data from CNN/DailyMail (no login needed)"
    )
    parser.add_argument("--cap", type=int, default=500,
                        help="Max rows per class (default: 500). "
                             "Note: Long class will be much smaller naturally.")
    parser.add_argument("--out", default="data/cnn_dailymail_test.csv")
    args = parser.parse_args()
    main(args.cap, args.out)
