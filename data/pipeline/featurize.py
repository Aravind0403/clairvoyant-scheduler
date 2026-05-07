import json
import os
import re
import csv
import random

from transformers import AutoTokenizer

# ── tokenizer: load once, reuse ──────────────────────────────────────────────
_TOKENIZER = None
_TOKENIZER_NAME = None

def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is not None:
        return _TOKENIZER
    for model_id in ("google/gemma-7b", "bert-base-uncased"):
        try:
            print(f"Loading tokenizer: {model_id} …")
            _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            _TOKENIZER_NAME = model_id
            print(f"  ✓ Loaded: {model_id}")
            return _TOKENIZER
        except Exception as exc:
            print(f"  ✗ {model_id} failed ({exc}), trying fallback…")
    raise RuntimeError("Could not load any tokenizer.")

def count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

# ── format keywords ───────────────────────────────────────────────────────────
FORMAT_KEYWORDS = [
    "list", "table", "bullet", "step by step", "enumerate",
    "outline", "numbered", "format", "structure",
]

def has_format_keyword(text: str) -> int:
    lower = text.lower()
    return int(any(kw in lower for kw in FORMAT_KEYWORDS))

def clause_count(text: str) -> int:
    return (
        text.count(',')
        + text.count(';')
        + text.count(' and ')
        + text.count(' but ')
        + text.count(' because ')
    )

# ── existing keyword / pattern sets ──────────────────────────────────────────
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

def extract_features(human_text, assistant_text):
    words = set(re.findall(r"\b\w+\b", human_text.lower()))

    instruction_verb = "other"
    for label, pattern in VERB_PATTERNS:
        if pattern.search(human_text):
            instruction_verb = label
            break

    human_clean = human_text.rstrip()
    ends_with_question = 1 if human_clean and human_clean.endswith("?") else 0

    return {
        "prompt_token_len":      count_tokens(human_text),
        "has_code_keyword":      int(bool(words & CODE_KEYWORDS)),
        "has_length_constraint": int(bool(CONSTRAINT_PATTERNS.search(human_text))),
        "instruction_verb":      instruction_verb,
        "ends_with_question":    ends_with_question,
        "has_format_keyword":    has_format_keyword(human_text),
        "clause_count":          clause_count(human_text),
        "actual_output_tokens":  count_tokens(assistant_text),
        "prompt":                human_text,
    }

def main():
    random.seed(42)
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path  = os.path.join(data_dir, "sharegpt_clean.json")
    out_path = os.path.join(data_dir, "training_data.csv")

    print(f"Loading {in_path}...")
    with open(in_path, "r") as f:
        data = json.load(f)

    # warm up tokenizer before the loop so progress is clear
    _get_tokenizer()

    shorts, mediums, longs = [], [], []

    for i, item in enumerate(data):
        if i % 5000 == 0 and i > 0:
            print(f"  processed {i:,} rows…")
        feats = extract_features(item['human_text'], item['assistant_text'])
        out_toks = feats['actual_output_tokens']

        if out_toks < 200:
            shorts.append(feats)
        elif out_toks < 800:
            mediums.append(feats)
        else:
            longs.append(feats)

    print(f"Initial split — Short={len(shorts)}, Medium={len(mediums)}, Long={len(longs)}")

    random.shuffle(shorts)
    random.shuffle(mediums)
    random.shuffle(longs)

    sampled_shorts  = shorts[:2000]
    sampled_mediums = mediums[:2000]
    sampled_longs   = longs[:2000]

    final_data = sampled_shorts + sampled_mediums + sampled_longs
    random.shuffle(final_data)

    features_keys = [
        "prompt",
        "prompt_token_len",
        "has_code_keyword",
        "has_length_constraint",
        "instruction_verb",
        "ends_with_question",
        "has_format_keyword",
        "clause_count",
        "actual_output_tokens",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=features_keys, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(final_data)

    print(f"Saved — Short={len(sampled_shorts)}, Medium={len(sampled_mediums)}, Long={len(sampled_longs)}")
    print(f"Total rows → {out_path}: {len(final_data)}")
    print(f"Tokenizer used: {_TOKENIZER_NAME}")

if __name__ == "__main__":
    main()
