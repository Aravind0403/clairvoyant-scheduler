"""
ablation.py — Drop-one feature ablation study for Clairvoyant

For each training dataset, drops one feature (or feature group) at a time,
retrains XGBoost on the reduced feature set, and reports ranking accuracy delta.

Feature groups:
  Numeric (6): prompt_token_len, has_code_keyword, has_length_constraint,
               ends_with_question, has_format_keyword, clause_count
  Verb group:  all 13 verb_* dummies (derived from instruction_verb)
               dropped together — they are one logical feature.

Ranking accuracy uses P(Long) as the continuous ranking score (same as train.py),
which is more precise than discrete class predictions.

Train/test split is fixed before the ablation loop so all feature-drop runs
compare against an identical held-out set. This eliminates split variance from
the delta measurement.

Usage:
    # All three training datasets (full ablation table):
    python model/ablation.py

    # Single dataset:
    python model/ablation.py --data data/lmsys_labeled.csv --label LMSYS

    # Save results to CSV for the paper:
    python model/ablation.py --out ablation_results.csv
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ── Feature schema — must match train.py exactly ─────────────────────────────

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

VERB_COLS = [f"verb_{v}" for v in KNOWN_VERBS]

LABEL_BINS  = [0, 200, 800, float("inf")]
LABEL_NAMES = {0: "Short", 1: "Medium", 2: "Long"}

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

# Feature groups for ablation.
# Each entry: (display_name, list_of_column_names_to_drop)
# The verb group drops all 13 verb_* columns in one shot.
ABLATION_GROUPS = [
    ("prompt_token_len",      ["prompt_token_len"]),
    ("has_code_keyword",      ["has_code_keyword"]),
    ("has_length_constraint", ["has_length_constraint"]),
    ("ends_with_question",    ["ends_with_question"]),
    ("has_format_keyword",    ["has_format_keyword"]),
    ("clause_count",          ["clause_count"]),
    ("instruction_verb (all verb_*)", VERB_COLS),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(csv_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV → (X: full feature matrix, y: integer labels)."""
    df = pd.read_csv(csv_path)

    y = pd.cut(
        df["actual_output_tokens"],
        bins=LABEL_BINS,
        labels=[0, 1, 2],
        right=False,
    ).astype(int)

    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )
    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=VERB_COLS, fill_value=0
    )

    present = [c for c in NUMERIC_COLS if c in df.columns]
    X = pd.concat([df[present].astype(float), verb_dummies], axis=1)
    return X, y


# ── Ranking accuracy (matches train.py) ──────────────────────────────────────

def ranking_accuracy(model: XGBClassifier,
                     X: pd.DataFrame,
                     y: pd.Series) -> float:
    """
    P(P_long[i] > P_long[j]) for all (i, j) pairs where actual_tokens[i] > actual_tokens[j].
    Uses continuous P(Long) scores — more discriminative than discrete class labels.
    """
    p_long    = model.predict_proba(X.values.astype(np.float32))[:, 2]
    idx_short = np.where(y.values == 0)[0]
    idx_long  = np.where(y.values == 2)[0]

    if not len(idx_short) or not len(idx_long):
        return float("nan")

    # Long should have higher P(Long) than Short
    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    total   = len(idx_long) * len(idx_short)
    return float(correct) / total


def classification_accuracy(model: XGBClassifier,
                             X: pd.DataFrame,
                             y: pd.Series) -> float:
    preds = model.predict(X.values.astype(np.float32))
    return float((preds == y.values).mean())


# ── Single ablation run ───────────────────────────────────────────────────────

def train_and_eval(X_train: pd.DataFrame,
                   X_test:  pd.DataFrame,
                   y_train: pd.Series,
                   y_test:  pd.Series,
                   drop_cols: list[str]) -> tuple[float, float]:
    """
    Drop the specified columns, train XGBoost, return (ranking_acc, class_acc).
    Columns absent from the feature matrix are silently ignored.
    """
    cols_to_drop = [c for c in drop_cols if c in X_train.columns]
    Xtr = X_train.drop(columns=cols_to_drop)
    Xte = X_test.drop(columns=cols_to_drop)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr, y_train)

    ra = ranking_accuracy(model, Xte, y_test)
    ca = classification_accuracy(model, Xte, y_test)
    return ra, ca


# ── Per-dataset ablation ──────────────────────────────────────────────────────

def run_ablation(csv_path: pathlib.Path,
                 label: str) -> list[dict]:
    """
    Run the full drop-one ablation for a single training dataset.
    Returns a list of result dicts (one per feature group + baseline).
    """
    print(f"\n{'═' * 68}")
    print(f"  Dataset: {label}  ({csv_path})")
    print(f"{'═' * 68}")

    X, y = load_features(csv_path)
    n_total = len(X)
    label_counts = y.value_counts().sort_index()
    print(f"  {n_total:,} rows · {X.shape[1]} features · "
          f"Short={label_counts.get(0,0)}  "
          f"Medium={label_counts.get(1,0)}  "
          f"Long={label_counts.get(2,0)}")

    # Fix the split once — all ablation runs use this identical held-out set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}  "
          f"(fixed split — seed 42)")

    # Baseline: all features
    print(f"\n  Training baseline (all {X.shape[1]} features)…")
    base_ra, base_ca = train_and_eval(X_train, X_test, y_train, y_test, drop_cols=[])

    results = [{
        "dataset":    label,
        "dropped":    "(none — baseline)",
        "n_features": X.shape[1],
        "ranking_acc": base_ra,
        "class_acc":   base_ca,
        "delta_ra":    0.0,
        "delta_ca":    0.0,
    }]

    print(f"  Baseline → Ranking: {base_ra*100:.2f}%  "
          f"Classification: {base_ca*100:.2f}%")

    # Drop-one runs
    print(f"\n  Running {len(ABLATION_GROUPS)} drop-one ablations…")
    ablation_results = []

    for feat_name, drop_cols in ABLATION_GROUPS:
        present = [c for c in drop_cols if c in X_train.columns]
        n_dropped = len(present)
        n_remaining = X.shape[1] - n_dropped

        ra, ca = train_and_eval(X_train, X_test, y_train, y_test, drop_cols=drop_cols)
        delta_ra = ra - base_ra
        delta_ca = ca - base_ca

        ablation_results.append({
            "dataset":    label,
            "dropped":    feat_name,
            "n_features": n_remaining,
            "ranking_acc": ra,
            "class_acc":   ca,
            "delta_ra":    delta_ra,
            "delta_ca":    delta_ca,
        })

    # Sort by ranking accuracy impact (most harmful drop first)
    ablation_results.sort(key=lambda r: r["delta_ra"])
    results.extend(ablation_results)

    # Print table
    col_w = 30
    print(f"\n  {'Feature dropped':<{col_w}}  {'Rank Acc':>9}  {'Δ Rank':>8}  "
          f"{'Class Acc':>10}  {'Δ Class':>8}  {'n_feat':>6}")
    print(f"  {'─' * (col_w + 48)}")

    for r in results:
        delta_ra_str = (f"{r['delta_ra']*100:+.2f}pp"
                        if r["dropped"] != "(none — baseline)" else "   —")
        delta_ca_str = (f"{r['delta_ca']*100:+.2f}pp"
                        if r["dropped"] != "(none — baseline)" else "   —")
        print(f"  {r['dropped']:<{col_w}}  "
              f"{r['ranking_acc']*100:>8.2f}%  "
              f"{delta_ra_str:>8}  "
              f"{r['class_acc']*100:>9.2f}%  "
              f"{delta_ca_str:>8}  "
              f"{r['n_features']:>6}")

    return results


# ── Summary across datasets ───────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    """
    Average delta_ra across datasets for each feature group.
    Reveals which features matter consistently vs dataset-specifically.
    """
    print(f"\n{'═' * 68}")
    print(f"  Summary — Average Ranking Accuracy Drop Across All Datasets")
    print(f"  (most harmful drop first — the feature that hurts most when removed)")
    print(f"{'═' * 68}")

    datasets = list(dict.fromkeys(r["dataset"] for r in all_results))
    features = [name for name, _ in ABLATION_GROUPS]

    # Collect baseline per dataset
    baselines = {}
    for r in all_results:
        if r["dropped"] == "(none — baseline)":
            baselines[r["dataset"]] = r["ranking_acc"]

    # Average delta per feature
    avg_deltas = []
    for feat_name in features:
        deltas = [r["delta_ra"] for r in all_results
                  if r["dropped"] == feat_name]
        if deltas:
            avg_deltas.append((feat_name, sum(deltas) / len(deltas), deltas))

    avg_deltas.sort(key=lambda x: x[1])

    print(f"\n  {'Feature':<35}  {'Avg Δ':>8}  " +
          "  ".join(f"{d:>10}" for d in datasets))
    print(f"  {'─' * (35 + 12 + 14 * len(datasets))}")

    for feat_name, avg_d, per_ds in avg_deltas:
        per_str = "  ".join(f"{d*100:>+9.2f}pp" for d in per_ds)
        print(f"  {feat_name:<35}  {avg_d*100:>+7.2f}pp  {per_str}")

    print(f"\n  Baselines:")
    for ds, b in baselines.items():
        print(f"    {ds}: {b*100:.2f}%")


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASET_REGISTRY = [
    ("ShareGPT", "data/training_data.csv"),
    ("LMSYS",    "data/lmsys_labeled.csv"),
    ("OASST1",   "data/oasst1_labeled.csv"),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drop-one feature ablation study for Clairvoyant"
    )
    parser.add_argument("--data",  default=None,
                        help="Path to a single training CSV (single-dataset mode)")
    parser.add_argument("--label", default="Dataset",
                        help="Label for --data (used in output)")
    parser.add_argument("--out",   default=None,
                        help="Optional: save all results to this CSV path")
    args = parser.parse_args()

    if args.data:
        datasets = [(args.label, args.data)]
    else:
        datasets = DATASET_REGISTRY

    all_results = []
    for label, path in datasets:
        p = pathlib.Path(path)
        if not p.exists():
            print(f"  [skip] {label} — {p} not found", file=sys.stderr)
            continue
        results = run_ablation(p, label)
        all_results.extend(results)

    if len(all_results) == 0:
        print("ERROR: no datasets found.", file=sys.stderr)
        sys.exit(1)

    if len(datasets) > 1 and len(all_results) > len(ABLATION_GROUPS) + 1:
        print_summary(all_results)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["dataset", "dropped", "n_features",
                      "ranking_acc", "class_acc", "delta_ra", "delta_ca"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
