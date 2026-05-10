"""
collect_lmsys_labeled.py — Build Model B training data from LMSYS-Chat-1M

Streams LMSYS-Chat-1M from HuggingFace (no full download).
Uses EXISTING response text — no Ollama inference needed.
Filters to small open-source models (similar size class to Gemma3:4b).
Counts response tokens with bert-base-uncased (same tokenizer as training pipeline).
Extracts the same 19 features as featurize.py.
Outputs a balanced CSV drop-in compatible with train.py.

Runtime: ~10-20 minutes on M1 (tokenizer + streaming, no GPU needed)

Usage:
    python data/pipeline/collect_lmsys_labeled.py \\
        [--n 6000]  \\
        [--cap 2000] \\
        [--out data/lmsys_labeled.csv]

Then retrain:
    python model/train.py \\
        --data data/lmsys_labeled.csv \\
        --out  model/predictor_model_b.json
"""

import argparse
import csv
import json
import os
import re
import random
import sys

# ── Models to EXCLUDE — large proprietary / 70B+ open models ─────────────────
# We want similar-sized models to Gemma3:4b (~4-7B range)
# Exclude GPT-4, Claude, PaLM, and 70B+ open models
EXCLUDED_MODEL_SUBSTRINGS = [
    "gpt-4", "gpt4",
    "gpt-3.5", "gpt3.5",
    "claude",
    "palm", "bard", "gemini",
    "text-davinci",
    "llama-2-70b", "llama-70b",
    "llama-3-70b", "llama3-70b",
    "falcon-40b", "falcon-180b",
    "vicuna-33b",
    "wizardlm-70b",
    "mistral-medium", "mistral-large",
    "yi-34b", "yi-6b-200k",
    "qwen-72b",
]

def is_small_model(model_name: str) -> bool:
    """True if model is small open-source (comparable to Gemma3:4b)."""
    name = model_name.lower()
    return not any(excl in name for excl in EXCLUDED_MODEL_SUBSTRINGS)


# ── Feature extraction — exact copy of featurize.py logic ────────────────────

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
        "prompt_token_len":      len(prompt) // 4,   # runtime approx, same as Go
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

    # Response token counting: use len//4 approximation.
    # bert-base-uncased truncates at 512 tokens — any Long response (800+ tokens)
    # gets miscounted as Medium, starving the Long class entirely.
    # len//4 matches the Go scheduler's runtime approximation and has no truncation limit.
    def count_response_tokens(text: str) -> int:
        return len(text) // 4

    # Prompt token len still uses bert for the prompt_token_len feature
    # (prompt lengths are short, well under 512, no truncation risk)
    try:
        from transformers import AutoTokenizer
        print("Loading bert-base-uncased tokenizer (for prompt features)...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("  ✓ Tokenizer loaded")
        def count_prompt_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))
    except ImportError:
        print("WARNING: transformers not found, using len//4 for prompt tokens too")
        def count_prompt_tokens(text: str) -> int:  # type: ignore
            return len(text) // 4

    print(f"\nStreaming LMSYS-Chat-1M (scanning up to {n_scan:,} rows)...")

    try:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "not found" in str(e).lower() or "access" in str(e).lower():
            print("\nERROR: LMSYS-Chat-1M is a gated dataset.", file=sys.stderr)
            print("Fix:", file=sys.stderr)
            print("  1. Visit https://huggingface.co/datasets/lmsys/lmsys-chat-1m", file=sys.stderr)
            print("     and accept the dataset terms (requires HF account)", file=sys.stderr)
            print("  2. Run: huggingface-cli login", file=sys.stderr)
            print("  3. Re-run this script", file=sys.stderr)
        else:
            print(f"\nERROR loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    buckets   = {0: [], 1: [], 2: []}   # Short / Medium / Long
    scanned   = 0
    kept      = 0
    filtered  = {"lang": 0, "model": 0, "turns": 0, "length": 0, "empty": 0}

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

        # ── filters ──────────────────────────────────────────────────────────
        if row.get("language", "English") != "English":
            filtered["lang"] += 1
            continue

        model_name = row.get("model", "")
        if not is_small_model(model_name):
            filtered["model"] += 1
            continue

        turns = row.get("conversation", [])
        if len(turns) < 2:
            filtered["turns"] += 1
            continue

        # LMSYS uses "user" / "assistant" (not "human" / "gpt")
        human_turn    = next((t for t in turns if t.get("role") in ("user", "human")), None)
        assistant_turn = next((t for t in turns if t.get("role") == "assistant"), None)

        if not human_turn or not assistant_turn:
            filtered["turns"] += 1
            continue

        prompt   = human_turn.get("content", "").strip()
        response = assistant_turn.get("content", "").strip()

        if not prompt or not response:
            filtered["empty"] += 1
            continue

        if not (10 <= len(prompt) <= 2000):
            filtered["length"] += 1
            continue

        # ── count tokens on existing response ────────────────────────────────
        resp_tokens = count_response_tokens(response)
        if resp_tokens == 0:
            filtered["empty"] += 1
            continue

        # ── label and bucket ─────────────────────────────────────────────────
        if resp_tokens < 200:
            lbl = 0
        elif resp_tokens < 800:
            lbl = 1
        else:
            lbl = 2

        if len(buckets[lbl]) >= cap:
            continue   # this class is full, keep scanning for others

        feats = extract_features(prompt, resp_tokens)
        buckets[lbl].append(feats)
        kept += 1

    # ── report ───────────────────────────────────────────────────────────────
    print(f"\nScanned: {scanned:,}  Kept: {kept:,}")
    print(f"Filter breakdown: {filtered}")
    for lbl, name in {0: "Short", 1: "Medium", 2: "Long"}.items():
        print(f"  {name}: {len(buckets[lbl]):,}")

    # ── balance to smallest class ─────────────────────────────────────────────
    random.seed(42)
    min_class   = min(len(b) for b in buckets.values())
    n_per_class = min(min_class, cap)

    if n_per_class == 0:
        print("\nERROR: one or more classes is empty.", file=sys.stderr)
        print("Try increasing --n or relaxing model filters.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for lbl in [0, 1, 2]:
        random.shuffle(buckets[lbl])
        rows.extend(buckets[lbl][:n_per_class])
    random.shuffle(rows)

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\nSaved {total:,} rows ({n_per_class} per class) → {out_path}")
    print(f"\nNext step:")
    print(f"  python model/train.py --data {out_path} --out model/predictor_model_b.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Model B training data from LMSYS-Chat-1M (no Ollama needed)"
    )
    parser.add_argument("--n",   type=int, default=100_000,
                        help="Max rows to scan from LMSYS (default: 100000)")
    parser.add_argument("--cap", type=int, default=2000,
                        help="Max rows per class (default: 2000)")
    parser.add_argument("--out", default="data/lmsys_labeled.csv")
    args = parser.parse_args()
    main(args.n, args.cap, args.out)
