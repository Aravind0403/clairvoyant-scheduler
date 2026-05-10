"""
evaluate_ranking.py — Cross-distribution ranking accuracy evaluation

Loads a trained XGBoost model and a test CSV, then computes:
  1. Classification accuracy (3-class: Short/Medium/Long)
  2. Ranking accuracy — the paper's primary metric
     P(predicted score of shorter job < predicted score of longer job)
     over all (i, j) pairs where actual_output_tokens[i] < actual_output_tokens[j]

Produces a single number per (model, test_dataset) cell in the
cross-distribution generalisation matrix.

Usage (single evaluation):
    python model/evaluate_ranking.py \\
        --model model/predictor_v3.json \\
        --test  data/dolly_labeled.csv

Usage (full matrix — all models vs all test datasets):
    python model/evaluate_ranking.py --matrix

The --matrix mode auto-discovers *.json models in model/ and *.csv test sets
in data/ matching the naming conventions below.

Model naming convention:
    model/predictor_v3.json          → ShareGPT (Model A)
    model/predictor_model_b.json     → LMSYS    (Model B)
    model/predictor_alpaca.json      → Alpaca   (Model C)
    model/predictor_code_alpaca.json → CodeAlpaca (Model D)

Test set naming convention:
    data/training_data.csv           → ShareGPT
    data/lmsys_labeled.csv           → LMSYS
    data/alpaca_labeled.csv          → Alpaca
    data/code_alpaca_labeled.csv     → CodeAlpaca
    data/dolly_labeled.csv           → Dolly
    data/cnn_dailymail_test.csv      → CNN/DailyMail
    data/wildchat_test.csv           → WildChat
"""

import argparse
import pathlib
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ── Feature schema — must match train.py exactly ─────────────────────────────

NUMERIC_COLS = [
    "prompt_token_len", "has_code_keyword", "has_length_constraint",
    "ends_with_question", "has_format_keyword", "clause_count",
]

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define", "other",
]

LABEL_NAMES = {0: "Short", 1: "Medium", 2: "Long"}


def build_features(csv_path: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (feature_matrix, actual_output_tokens_array)."""
    df = pd.read_csv(csv_path)
    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )
    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=[f"verb_{v}" for v in KNOWN_VERBS], fill_value=0
    )
    present = [c for c in NUMERIC_COLS if c in df.columns]
    X = pd.concat([df[present].astype(float), verb_dummies], axis=1)
    y = df["actual_output_tokens"].values.astype(int)
    return X, y


def load_model(model_path: pathlib.Path) -> XGBClassifier:
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    return clf


def predict_scores(clf: XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (p_long, discrete_labels).
    p_long         : P(Long) — continuous score used for ranking comparison.
    discrete_labels: 0=Short, 1=Medium, 2=Long — used for classification accuracy.
    """
    X_np = X.values.astype(np.float32)
    p_long  = clf.predict_proba(X_np)[:, 2]
    labels  = clf.predict(X_np).astype(int)
    return p_long, labels


def ranking_accuracy_fast(p_long: np.ndarray, actual_tokens: np.ndarray) -> float:
    """
    Ranking accuracy over Short–Long pairs only (matches train.py).

    Metric: P( P_long[long_j] > P_long[short_i] )
    over all (i, j) pairs where actual_tokens[i] < 200 and actual_tokens[j] >= 800.

    Rationale: SJF scheduling cares about Long-before-Short violations.
    Short–Medium and Medium–Long ordering errors are less critical and
    would dilute the metric with noise from near-boundary examples.

    Consistent with train.py's ranking_accuracy function.
    """
    idx_short = np.where(actual_tokens < 200)[0]
    idx_long  = np.where(actual_tokens >= 800)[0]

    if not len(idx_short) or not len(idx_long):
        return float("nan")

    # (n_long, n_short) matrix: True where Long has higher P(Long) than Short
    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    total   = len(idx_long) * len(idx_short)
    return float(correct) / total


def classification_accuracy(pred_labels: np.ndarray, actual_tokens: np.ndarray) -> float:
    """Standard 3-class accuracy: compare predicted class to ground-truth class."""
    def to_class(t):
        if t < 200:  return 0
        if t < 800:  return 1
        return 2
    true_labels = np.array([to_class(t) for t in actual_tokens])
    return (pred_labels == true_labels).mean()


def per_class_accuracy(pred_labels: np.ndarray, actual_tokens: np.ndarray) -> dict:
    def to_class(t):
        if t < 200:  return 0
        if t < 800:  return 1
        return 2
    true_labels = np.array([to_class(t) for t in actual_tokens])
    result = {}
    for cls in [0, 1, 2]:
        mask = true_labels == cls
        if mask.sum() == 0:
            result[LABEL_NAMES[cls]] = float("nan")
        else:
            result[LABEL_NAMES[cls]] = (pred_labels[mask] == cls).mean()
    return result


# ── Model/dataset registry for --matrix mode ─────────────────────────────────

MODEL_REGISTRY = {
    "ShareGPT":  "model/predictor_v3.json",
    "LMSYS":     "model/predictor_model_b.json",
    "OASST1":    "model/predictor_oasst1.json",
    # Alpaca / CodeAlpaca intentionally excluded:
    # both have <5 Long examples → training on 9–12 rows produces overfit,
    # degenerate models that predict everything as one class.
    # Finding: curated instruction datasets are Short-biased and cannot serve
    # as SJF training sources. See paper §4 (Dataset Selection).
}

DATASET_REGISTRY = {
    "ShareGPT":      "data/training_data.csv",
    "LMSYS":         "data/lmsys_labeled.csv",
    "OASST1":        "data/oasst1_labeled.csv",
    "Dolly":         "data/dolly_labeled.csv",
    "CNN/DailyMail": "data/cnn_dailymail_test.csv",
    # Alpaca / CodeAlpaca as test columns: ranking accuracy is unreliable
    # (<5 Long examples → high-variance metric). Excluded from matrix.
    # WildChat requires HF login — add manually if collected.
    # "WildChat":    "data/wildchat_test.csv",
}


def run_single(model_path: pathlib.Path, test_path: pathlib.Path, verbose: bool = True):
    """Evaluate one (model, test_dataset) cell. Returns (ranking_acc, class_acc)."""
    clf     = load_model(model_path)
    X, y    = build_features(test_path)
    scores, labels = predict_scores(clf, X)

    rank_acc  = ranking_accuracy_fast(scores, y)
    class_acc = classification_accuracy(labels, y)

    if verbose:
        print(f"\nModel      : {model_path}")
        print(f"Test set   : {test_path}  ({len(y):,} rows)")
        print(f"\nRanking accuracy  : {rank_acc*100:.2f}%")
        print(f"Classification acc: {class_acc*100:.2f}%")
        print(f"\nPer-class accuracy:")
        for name, acc in per_class_accuracy(labels, y).items():
            print(f"  {name:<8}: {acc*100:.1f}%")

    return rank_acc, class_acc


def run_matrix(base_dir: pathlib.Path):
    """Build the full cross-distribution generalisation matrix."""
    print("\n" + "═" * 72)
    print("  Cross-Distribution Generalisation Matrix")
    print("  Metric: Ranking Accuracy (%)")
    print("═" * 72)

    available_models  = {}
    available_datasets = {}

    for name, path in MODEL_REGISTRY.items():
        p = base_dir / path
        if p.exists():
            available_models[name] = p
        else:
            print(f"  [skip model] {name} — {p} not found")

    for name, path in DATASET_REGISTRY.items():
        p = base_dir / path
        if p.exists():
            available_datasets[name] = p
        else:
            print(f"  [skip test]  {name} — {p} not found")

    if not available_models:
        print("\nERROR: no models found. Train at least one model first.", file=sys.stderr)
        sys.exit(1)

    if not available_datasets:
        print("\nERROR: no test datasets found.", file=sys.stderr)
        sys.exit(1)

    # Header row
    col_names = list(available_datasets.keys())
    col_w     = 14
    header    = f"{'Train ↓ / Test →':<20}" + "".join(f"{c:>{col_w}}" for c in col_names)
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    # Results storage for summary
    results = {}

    for model_name, model_path in available_models.items():
        clf = load_model(model_path)
        row_results = {}

        row_str = f"{model_name:<20}"
        for ds_name, ds_path in available_datasets.items():
            try:
                X, y   = build_features(ds_path)
                scores, _ = predict_scores(clf, X)
                ra     = ranking_accuracy_fast(scores, y)
                cell   = f"{ra*100:.1f}%" if not np.isnan(ra) else "  n/a"
                row_results[ds_name] = ra
            except Exception as e:
                cell = " err"
                row_results[ds_name] = float("nan")
                print(f"    [WARN] {model_name} × {ds_name}: {e}", file=sys.stderr)

            row_str += f"{cell:>{col_w}}"

        print(row_str)
        results[model_name] = row_results

    print("\n" + "─" * 72)
    print("  Diagonal = in-distribution accuracy")
    print("  Off-diagonal = cross-distribution generalisation")
    print("  Test-only columns (Dolly, CNN/DailyMail, WildChat) have no diagonal entry")

    # Print classification accuracy matrix too
    print("\n" + "═" * 72)
    print("  Classification Accuracy Matrix (3-class)")
    print("═" * 72)
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for model_name, model_path in available_models.items():
        clf = load_model(model_path)
        row_str = f"{model_name:<20}"
        for ds_name, ds_path in available_datasets.items():
            try:
                X, y   = build_features(ds_path)
                _, labels = predict_scores(clf, X)
                ca     = classification_accuracy(labels, y)
                cell   = f"{ca*100:.1f}%"
            except Exception:
                cell   = " err"
            row_str += f"{cell:>{col_w}}"
        print(row_str)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-distribution ranking accuracy evaluation for Clairvoyant"
    )
    parser.add_argument("--model", default=None,
                        help="Path to XGBoost .json model (single-cell mode)")
    parser.add_argument("--test",  default=None,
                        help="Path to test CSV (single-cell mode)")
    parser.add_argument("--matrix", action="store_true",
                        help="Build the full cross-distribution matrix")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory for --matrix mode (default: .)")
    args = parser.parse_args()

    if args.matrix:
        run_matrix(pathlib.Path(args.base_dir))
    elif args.model and args.test:
        run_single(pathlib.Path(args.model), pathlib.Path(args.test))
    else:
        parser.print_help()
        print("\nERROR: provide --model + --test for single cell, or --matrix for full matrix.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
