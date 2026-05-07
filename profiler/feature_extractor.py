"""Extract the 5 prompt features used as training inputs.

Token counting
--------------
Uses the fast approximation  len(text) // 4  (≈ 1 token per 4 chars),
which is sufficient for relative comparisons and requires no dependencies.

Features
--------
prompt_token_len      : int  – approximate token count of the prompt
has_code_keyword      : 0|1  – prompt contains code-related keywords
has_length_constraint : 0|1  – prompt contains an explicit length constraint
instruction_verb      : str  – dominant task verb (write/explain/list/…/other)
ends_with_question    : 0|1  – prompt's last non-whitespace char is '?'
"""

import re
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Approximate token count: len(text) // 4."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = frozenset([
    "code", "function", "def", "class", "script", "program",
    "implement", "algorithm", "debug", "refactor", "import",
    "variable", "loop", "recursion", "api", "sql", "regex",
])

_CONSTRAINT_PATTERNS = re.compile(
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

# Ordered — first match wins
_VERB_PATTERNS: list[tuple[str, re.Pattern]] = [
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


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class Features:
    prompt_token_len: int
    has_code_keyword: int
    has_length_constraint: int
    instruction_verb: str
    ends_with_question: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract(prompt: str) -> Features:
    words = set(re.findall(r"\b\w+\b", prompt.lower()))

    instruction_verb = "other"
    for label, pattern in _VERB_PATTERNS:
        if pattern.search(prompt):
            instruction_verb = label
            break

    return Features(
        prompt_token_len=approx_tokens(prompt),
        has_code_keyword=int(bool(words & _CODE_KEYWORDS)),
        has_length_constraint=int(bool(_CONSTRAINT_PATTERNS.search(prompt))),
        instruction_verb=instruction_verb,
        ends_with_question=int(prompt.rstrip().endswith("?")),
    )
