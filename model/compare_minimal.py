"""
compare_minimal.py — Full vs Minimal feature set cross-distribution comparison

Ablation study showed that 3 features are net-harmful (positive avg Δ when dropped):
  has_length_constraint  −0.12pp avg (marginal, noisy)
  has_format_keyword     +0.78pp avg (actively harmful)
  clause_count           +1.07pp avg (actively harmful)

This script retrains all 3 models with the MINIMAL feature set
(16 features: 3 numeric + 13 verb dummies) and runs the full
cross-distribution matrix for both Full and Minimal models side-by-side.

Minimal feature set:
  Numeric (3): prompt_token_len, has_code_keyword, ends_with_question
  One-hot (13): instruction_verb → 13 verb dummies
  DROPPED: has_length_constraint, has_format_keyword, clause_count

Usage:
    python model/compare_minimal.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ── Feature schema ────────────────────────────────────────────────────────────

FULL_NUMERIC = [
    "prompt_token_len", "has_code_keyword", "has_length_constraint",
    "ends_with_question", "has_format_keyword", "clause_count",
]

MINIMAL_NUMERIC = [
    "prompt_token_len", "has_code_keyword", "ends_with_question",
]

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define", "other",
]
VERB_COLS = [f"verb_{v}" for v in KNOWN_VERBS]

LABEL_BINS = [0, 200, 800, float("inf")]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

TRAIN_DATASETS = [
    ("ShareGPT", "data/training_data.csv"),
    ("LMSYS",    "data/lmsys_labeled.csv"),
    ("OASST1",   "data/oasst1_labeled.csv"),
]

TEST_DATASETS = [
    ("ShareGPT",      "data/training_data.csv"),
    ("LMSYS",         "data/lmsys_labeled.csv"),
    ("OASST1",        "data/oasst1_labeled.csv"),
    ("Dolly",         "data/dolly_labeled.csv"),
    ("CNN/DailyMail", "data/cnn_dailymail_test.csv"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(csv_path: pathlib.Path,
                  numeric_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)

    y = pd.cut(
        df["actual_output_tokens"],
        bins=LABEL_BINS,
        labels=[0, 1, 2],
        right=False,
    ).astype(int).values

    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )
    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=VERB_COLS, fill_value=0
    )
    present = [c for c in numeric_cols if c in df.columns]
    X = pd.concat([df[present].astype(float), verb_dummies], axis=1)
    return X, y


# ── Ranking accuracy (P(Long) continuous score — same as train.py) ────────────

def ranking_accuracy(model: XGBClassifier,
                     X: pd.DataFrame,
                     y: np.ndarray) -> float:
    p_long    = model.predict_proba(X.values.astype(np.float32))[:, 2]
    idx_short = np.where(y == 0)[0]
    idx_long  = np.where(y == 2)[0]
    if not len(idx_short) or not len(idx_long):
        return float("nan")
    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    return float(correct) / (len(idx_long) * len(idx_short))


# ── Train one model, return it (no saving to disk) ────────────────────────────

def train_model(csv_path: pathlib.Path, numeric_cols: list[str]) -> XGBClassifier:
    X, y = load_features(csv_path, numeric_cols)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42,
        stratify=y,
    )
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


# ── Evaluate one trained model against one test CSV ───────────────────────────

def eval_model(model: XGBClassifier,
               test_path: pathlib.Path,
               numeric_cols: list[str]) -> float:
    X, y = load_features(test_path, numeric_cols)
    return ranking_accuracy(model, X, y)


# ── Print a ranking matrix ────────────────────────────────────────────────────

def print_matrix(title: str,
                 matrix: dict,          # {train_label: {test_label: float}}
                 train_labels: list[str],
                 test_labels: list[str]) -> None:
    col_w = 14
    print(f"\n{'═' * (22 + col_w * len(test_labels))}")
    print(f"  {title}")
    print(f"{'═' * (22 + col_w * len(test_labels))}")
    header = f"{'Train ↓ / Test →':<22}" + "".join(f"{t:>{col_w}}" for t in test_labels)
    print(f"\n{header}")
    print(f"  {'─' * (len(header) - 2)}")
    for train_lbl in train_labels:
        row = f"{train_lbl:<22}"
        for test_lbl in test_labels:
            v = matrix.get(train_lbl, {}).get(test_lbl, float("nan"))
            cell = f"{v*100:.1f}%" if not np.isnan(v) else "  n/a"
            row += f"{cell:>{col_w}}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = pathlib.Path(".")

    # Validate all paths exist
    missing = []
    for label, path in TRAIN_DATASETS + TEST_DATASETS:
        if not (base / path).exists():
            missing.append(f"  {label}: {path}")
    if missing:
        print("Missing datasets (run collectors first):", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        sys.exit(1)

    train_labels = [l for l, _ in TRAIN_DATASETS]
    test_labels  = [l for l, _ in TEST_DATASETS]

    full_matrix    = {l: {} for l in train_labels}
    minimal_matrix = {l: {} for l in train_labels}
    delta_matrix   = {l: {} for l in train_labels}

    for train_label, train_path in TRAIN_DATASETS:
        tp = base / train_path
        print(f"\nTraining Full    model on {train_label}…", end=" ", flush=True)
        full_model = train_model(tp, FULL_NUMERIC)
        print("done")

        print(f"Training Minimal model on {train_label}…", end=" ", flush=True)
        minimal_model = train_model(tp, MINIMAL_NUMERIC)
        print("done")

        for test_label, test_path in TEST_DATASETS:
            tsp = base / test_path
            full_ra    = eval_model(full_model,    tsp, FULL_NUMERIC)
            minimal_ra = eval_model(minimal_model, tsp, MINIMAL_NUMERIC)
            full_matrix[train_label][test_label]    = full_ra
            minimal_matrix[train_label][test_label] = minimal_ra
            delta_matrix[train_label][test_label]   = minimal_ra - full_ra

    # ── Print matrices ────────────────────────────────────────────────────────
    print_matrix("Full Model (19 features) — Ranking Accuracy",
                 full_matrix, train_labels, test_labels)

    print_matrix("Minimal Model (16 features, dropped 3 harmful) — Ranking Accuracy",
                 minimal_matrix, train_labels, test_labels)

    # Delta matrix: positive = minimal is better
    col_w = 14
    print(f"\n{'═' * (22 + col_w * len(test_labels))}")
    print(f"  Delta: Minimal − Full  (positive = minimal wins)")
    print(f"{'═' * (22 + col_w * len(test_labels))}")
    header = f"{'Train ↓ / Test →':<22}" + "".join(f"{t:>{col_w}}" for t in test_labels)
    print(f"\n{header}")
    print(f"  {'─' * (len(header) - 2)}")
    for train_lbl in train_labels:
        row = f"{train_lbl:<22}"
        for test_lbl in test_labels:
            v = delta_matrix[train_lbl][test_lbl]
            cell = f"{v*100:+.1f}pp" if not np.isnan(v) else "  n/a"
            row += f"{cell:>{col_w}}"
        print(row)

    # ── Average deltas ────────────────────────────────────────────────────────
    all_deltas = [
        delta_matrix[trl][tel]
        for trl in train_labels
        for tel in test_labels
        if not np.isnan(delta_matrix[trl][tel])
    ]
    avg_delta = sum(all_deltas) / len(all_deltas) if all_deltas else float("nan")

    diag_deltas = [
        delta_matrix[l][l]
        for l in train_labels
        if l in {tl for tl, _ in TEST_DATASETS}
        and not np.isnan(delta_matrix[l].get(l, float("nan")))
    ]
    avg_diag_delta = sum(diag_deltas) / len(diag_deltas) if diag_deltas else float("nan")

    off_deltas = [
        delta_matrix[trl][tel]
        for trl in train_labels
        for tel in test_labels
        if trl != tel and not np.isnan(delta_matrix[trl][tel])
    ]
    avg_off_delta = sum(off_deltas) / len(off_deltas) if off_deltas else float("nan")

    print(f"\n{'─' * 60}")
    print(f"  Average delta (all cells)       : {avg_delta*100:+.2f}pp")
    print(f"  Average delta (in-distribution) : {avg_diag_delta*100:+.2f}pp")
    print(f"  Average delta (cross-dist)      : {avg_off_delta*100:+.2f}pp")
    print(f"\n  Interpretation:")
    print(f"    +pp avg = minimal model generalises better (drop the 3 features)")
    print(f"    −pp avg = full model wins (keep all 19 features)")


if __name__ == "__main__":
    main()
