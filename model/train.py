"""
Clairvoyant Scheduler — XGBoost complexity classifier (v3)

Labels
------
  0 = Short   actual_output_tokens <  200
  1 = Medium  actual_output_tokens  200 – 799
  2 = Long    actual_output_tokens >= 800

Features
--------
  Numeric : prompt_token_len, has_code_keyword, has_length_constraint,
            ends_with_question, has_format_keyword, clause_count   ← v3 additions
  One-hot : instruction_verb → 13 known verbs (anything else → "other")

Usage
-----
  python model/train.py [--data data/training_data.csv]
                        [--out  model/predictor_v3.json]
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LABEL_BINS  = [0, 200, 800, float("inf")]
LABEL_NAMES = ["Short", "Medium", "Long"]

NUMERIC_COLS = [
    "prompt_token_len",
    "has_code_keyword",
    "has_length_constraint",
    "ends_with_question",
    "has_format_keyword",   # v3
    "clause_count",         # v3
]

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define",
    "other",
]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    df["label"] = pd.cut(
        df["actual_output_tokens"],
        bins=LABEL_BINS,
        labels=[0, 1, 2],
        right=False,
    ).astype(int)

    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )

    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=[f"verb_{v}" for v in KNOWN_VERBS], fill_value=0
    )

    # only include NUMERIC_COLS that are actually present (graceful v2→v3 migration)
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    X = pd.concat([df[present_numeric].astype(float), verb_dummies], axis=1)
    y = df["label"]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ranking_accuracy(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """Fraction of (Short, Long) test pairs correctly ordered by P(Long)."""
    p_long    = model.predict_proba(X)[:, 2]
    idx_short = np.where(y.values == 0)[0]
    idx_long  = np.where(y.values == 2)[0]

    if not len(idx_short) or not len(idx_long):
        return float("nan")

    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    total   = len(idx_long) * len(idx_short)
    return float(correct) / total


def _section(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: pathlib.Path, out_path: pathlib.Path) -> None:

    print(f"Loading  {csv_path} …")
    X, y = load_and_prepare(csv_path)
    label_counts = y.value_counts().sort_index().to_dict()
    print(f"  {len(X):,} rows · {X.shape[1]} features · "
          f"labels {{ {', '.join(f'{LABEL_NAMES[k]}={v}' for k, v in label_counts.items())} }}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    _section("Training XGBoost")
    for k, v in XGB_PARAMS.items():
        print(f"  {k:<25} = {v}")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print("\n  Training complete.")

    y_pred = model.predict(X_test)

    _section("Classification Report")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=3))

    _section("Confusion Matrix  (rows = actual · cols = predicted)")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    w  = 10
    print(" " * w + "".join(f"{n:>{w}}" for n in LABEL_NAMES))
    for i, row_name in enumerate(LABEL_NAMES):
        print(f"{row_name:>{w}}" + "".join(f"{cm[i, j]:>{w}}" for j in range(3)))

    _section("Top 5 Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top5 = importances.nlargest(5)
    for feat, score in top5.items():
        bar = "█" * max(1, int(score * 80))
        print(f"  {feat:<35} {score:.4f}  {bar}")

    _section("Ranking Accuracy  (Short < Long pairs)")
    ra      = ranking_accuracy(model, X_test, y_test)
    n_short = int((y_test == 0).sum())
    n_long  = int((y_test == 2).sum())
    print(f"  Pairs evaluated  : {n_short * n_long:,}  ({n_short} Short × {n_long} Long)")
    print(f"  Correctly ordered: {ra * 100:.2f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    _section("Model Saved")
    print(f"  {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clairvoyant Scheduler — train v3")
    parser.add_argument("--data", default="data/training_data.csv")
    parser.add_argument("--out",  default="model/predictor_v3.json")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.data)
    if not csv_path.exists():
        print(f"ERROR: data file not found — {csv_path}", file=sys.stderr)
        sys.exit(1)

    main(csv_path=csv_path, out_path=pathlib.Path(args.out))
