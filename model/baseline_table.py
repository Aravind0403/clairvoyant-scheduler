"""
Baseline comparison table for Clairvoyant paper.

Compares four scheduling approaches on pairwise ranking accuracy
(P(score_long[Long] > score_long[Short])) across three datasets.

Methods:
  1. FCFS (random)         — 50% baseline
  2. Prompt-length rule    — predict Long if prompt_token_len > threshold
  3. Keyword heuristic     — predict Long if has_code_keyword == 1
  4. Clairvoyant (XGBoost) — 19-feature trained model

Usage:
    python model/baseline_table.py
"""

import pathlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_BINS  = [0, 200, 800, float("inf")]
LABEL_NAMES = ["Short", "Medium", "Long"]

NUMERIC_COLS = [
    "prompt_token_len",
    "has_code_keyword",
    "has_length_constraint",
    "ends_with_question",
    "has_format_keyword",
    "clause_count",
]

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define", "other",
]

DATASETS = {
    "ShareGPT":   "data/training_data.csv",
    "LMSYS":      "data/lmsys_labeled.csv",
    "OASST1":     "data/oasst1_labeled.csv",
}

ROOT = pathlib.Path(__file__).parent.parent

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    verb = df["instruction_verb"].str.lower().str.strip()
    verb = verb.where(verb.isin(KNOWN_VERBS), other="other")
    dummies = pd.get_dummies(verb, prefix="verb")
    for v in KNOWN_VERBS:
        col = f"verb_{v}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"verb_{v}" for v in KNOWN_VERBS]]
    return pd.concat([df[NUMERIC_COLS].reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = pd.cut(df["actual_output_tokens"],
                         bins=LABEL_BINS, labels=[0, 1, 2])
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


# ── Ranking accuracy ──────────────────────────────────────────────────────────

def ranking_accuracy(scores_long: np.ndarray, labels: np.ndarray) -> float:
    """
    P(score_long[Long] > score_long[Short]) over all Short-Long pairs.
    Only Short (0) and Long (2) examples used.
    """
    short_mask = labels == 0
    long_mask  = labels == 2

    short_scores = scores_long[short_mask]
    long_scores  = scores_long[long_mask]

    if len(short_scores) == 0 or len(long_scores) == 0:
        return float("nan")

    correct = np.sum(
        long_scores[:, None] > short_scores[None, :]
    )
    total = len(long_scores) * len(short_scores)
    return correct / total


# ── Methods ───────────────────────────────────────────────────────────────────

def score_prompt_length(df_test: pd.DataFrame, threshold: int) -> np.ndarray:
    """Score = prompt_token_len / max (normalised to [0,1])."""
    vals = df_test["prompt_token_len"].values.astype(float)
    return vals / (vals.max() + 1e-9)


def score_keyword(df_test: pd.DataFrame) -> np.ndarray:
    """Score = has_code_keyword (binary)."""
    return df_test["has_code_keyword"].values.astype(float)


def score_xgboost(X_train, y_train, X_test) -> np.ndarray:
    """Train XGBoost on train split, return P(Long) on test split."""
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)  # shape (n, 3)
    return proba[:, 2]  # P(Long)


def best_threshold(df_train: pd.DataFrame, labels_train: np.ndarray) -> int:
    """Find prompt_token_len threshold that maximises ranking accuracy on train."""
    best_acc, best_t = 0.0, 20
    for t in range(5, 200, 5):
        scores = (df_train["prompt_token_len"].values > t).astype(float)
        acc = ranking_accuracy(scores, labels_train)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_dataset(name: str, path: str) -> dict:
    df = pd.read_csv(ROOT / path)
    df = add_labels(df)
    df = df.dropna(subset=["actual_output_tokens", "prompt_token_len",
                            "has_code_keyword", "instruction_verb"])

    X = build_features(df)
    y = df["label"].values

    # 80/20 split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    # Filter test to Short + Long only (for ranking accuracy)
    sl_mask = (y_test == 0) | (y_test == 2)
    X_test_sl  = X_test[sl_mask]
    y_test_sl  = y_test[sl_mask]
    df_test_sl = df_test[sl_mask]

    n_short = (y_test_sl == 0).sum()
    n_long  = (y_test_sl == 2).sum()

    # 1. FCFS (random)
    ra_fcfs = 0.50

    # 2. Prompt-length threshold (optimised on train)
    sl_train_mask = (y_train == 0) | (y_train == 2)
    threshold = best_threshold(df_train[sl_train_mask], y_train[sl_train_mask])
    scores_len = score_prompt_length(df_test_sl, threshold)
    ra_len = ranking_accuracy(scores_len, y_test_sl)

    # 3. Keyword heuristic
    scores_kw = score_keyword(df_test_sl)
    ra_kw = ranking_accuracy(scores_kw, y_test_sl)

    # 4. Clairvoyant (XGBoost)
    scores_xgb = score_xgboost(X_train, y_train, X_test_sl)
    ra_xgb = ranking_accuracy(scores_xgb, y_test_sl)

    print(f"\n{name}  (n_short={n_short}, n_long={n_long}, threshold={threshold})")
    print(f"  FCFS (random)          : {ra_fcfs*100:.1f}%")
    print(f"  Prompt-length rule     : {ra_len*100:.1f}%")
    print(f"  Keyword heuristic      : {ra_kw*100:.1f}%")
    print(f"  Clairvoyant (XGBoost)  : {ra_xgb*100:.1f}%")

    return {
        "dataset":       name,
        "n_short":       n_short,
        "n_long":        n_long,
        "threshold":     threshold,
        "fcfs":          round(ra_fcfs * 100, 1),
        "prompt_length": round(ra_len  * 100, 1),
        "keyword":       round(ra_kw   * 100, 1),
        "clairvoyant":   round(ra_xgb  * 100, 1),
    }


def main():
    print("Baseline comparison — pairwise ranking accuracy (Short vs Long pairs)")
    print("=" * 65)

    rows = []
    for name, path in DATASETS.items():
        full = ROOT / path
        if not full.exists():
            print(f"\n[SKIP] {name} — file not found: {full}")
            continue
        rows.append(evaluate_dataset(name, path))

    print("\n\nSUMMARY TABLE")
    print("=" * 65)
    print(f"{'Method':<26} {'ShareGPT':>10} {'LMSYS':>8} {'OASST1':>8}")
    print("-" * 55)

    methods = [
        ("FCFS (random)",         "fcfs"),
        ("Prompt-length rule",    "prompt_length"),
        ("Keyword heuristic",     "keyword"),
        ("Clairvoyant (XGBoost)", "clairvoyant"),
    ]

    for label, key in methods:
        vals = [str(r[key]) + "%" if key in r else "—" for r in rows]
        print(f"{label:<26} {vals[0]:>10} {vals[1]:>8} {vals[2]:>8}")

    print()


if __name__ == "__main__":
    main()
