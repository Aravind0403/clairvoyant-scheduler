"""
domain_retrain_ablation.py — Sample-size ablation for domain retraining (TODO-C)

Answers: at what N production samples does retraining converge?

Design
------
  Source  : data/serving_logs.jsonl  (captured via run_serving_logs.py)
  Sizes   : 500, 1000, 2000, 4000 total samples (balanced per class)
  Seeds   : 20 per cell (different train/test splits)
  Metric  : ranking accuracy (Short vs Long pairs, same as paper)
  Baseline: Model B (LMSYS predictor) evaluated on same test sets

Output
------
  results/domain_retrain_ablation.json   — full per-seed results
  results/domain_retrain_ablation.csv    — summary table for paper (Tab. format)

Usage
-----
  python model/domain_retrain_ablation.py
  python model/domain_retrain_ablation.py --sizes 500 1000  # if partial data only
  python model/domain_retrain_ablation.py --seeds 5         # quick smoke-test
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ── Label / feature config (must match train.py exactly) ─────────────────────

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

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1,
)

DEFAULT_SIZES = [500, 1000, 2000, 4000]
DEFAULT_SEEDS = 20


# ── Data loading ─────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["label"] = pd.cut(
        df["actual_output_tokens"],
        bins=LABEL_BINS, labels=[0, 1, 2], right=False,
    ).astype(int)

    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )
    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=[f"verb_{v}" for v in KNOWN_VERBS], fill_value=0
    )
    present = [c for c in NUMERIC_COLS if c in df.columns]
    X = pd.concat([df[present].astype(float), verb_dummies], axis=1)
    return X, df["label"]


def load_serving_logs(path: pathlib.Path) -> pd.DataFrame:
    """Load featurized serving logs CSV (output of featurize_serving_logs.py)."""
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        print("Run: python data/pipeline/featurize_serving_logs.py --cap 1500", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded domain data: {len(df):,} rows from {path}")
    label_col = pd.cut(df["actual_output_tokens"], bins=LABEL_BINS,
                       labels=["Short", "Medium", "Long"], right=False)
    print("  Class distribution:", label_col.value_counts().to_dict())
    return df


def load_model_b(path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load LMSYS training data for Model B baseline."""
    df = pd.read_csv(path)
    return prepare_features(df)


# ── Ranking accuracy ──────────────────────────────────────────────────────────

def ranking_accuracy(model: XGBClassifier,
                     X: pd.DataFrame, y: pd.Series) -> float:
    p_long    = model.predict_proba(X)[:, 2]
    idx_short = np.where(y.values == 0)[0]
    idx_long  = np.where(y.values == 2)[0]
    if not len(idx_short) or not len(idx_long):
        return float("nan")
    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    return float(correct) / (len(idx_long) * len(idx_short))


# ── Ablation core ─────────────────────────────────────────────────────────────

def run_ablation(domain_df: pd.DataFrame,
                 model_b_lmsys_path: pathlib.Path,
                 sizes: list[int],
                 n_seeds: int) -> dict:
    results = {}

    # Filter to only Short (0) and Long (2) for balanced sampling
    domain_df["_label"] = pd.cut(
        domain_df["actual_output_tokens"],
        bins=LABEL_BINS, labels=[0, 1, 2], right=False,
    ).astype(int)

    # Check max feasible size
    per_class_available = domain_df["_label"].value_counts().min()
    max_per_class = per_class_available
    max_total = max_per_class * 3
    print(f"\nDomain data: {per_class_available} samples in smallest class "
          f"→ max balanced total = {max_total}")

    feasible_sizes = [s for s in sizes if s <= max_total]
    skipped = [s for s in sizes if s > max_total]
    if skipped:
        print(f"Skipping sizes {skipped} — insufficient data "
              f"(collect more serving logs to enable these)")

    for total_n in feasible_sizes:
        n_per_class = total_n // 3
        print(f"\n{'─'*60}")
        print(f"  N = {total_n} ({n_per_class} per class), {n_seeds} seeds")
        print(f"{'─'*60}")

        seed_results = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)

            # Sample balanced subset from domain data
            sampled_parts = []
            for cls in [0, 1, 2]:
                cls_df = domain_df[domain_df["_label"] == cls]
                idx = rng.choice(len(cls_df), size=n_per_class, replace=False)
                sampled_parts.append(cls_df.iloc[idx])
            sample_df = pd.concat(sampled_parts).sample(frac=1, random_state=seed)

            X, y = prepare_features(sample_df)

            # 80/20 split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y
            )

            model = XGBClassifier(**XGB_PARAMS, random_state=seed)
            model.fit(X_train, y_train)
            ra = ranking_accuracy(model, X_test, y_test)
            seed_results.append(ra)

            if (seed + 1) % 5 == 0:
                so_far = np.array(seed_results)
                print(f"    seed {seed+1:2d}/{n_seeds}  "
                      f"running mean={so_far.mean()*100:.1f}%  "
                      f"std={so_far.std()*100:.1f}%")

        arr = np.array(seed_results)
        summary = {
            "n_total":    total_n,
            "n_per_class": n_per_class,
            "n_seeds":    n_seeds,
            "mean_pct":   round(float(arr.mean() * 100), 2),
            "std_pct":    round(float(arr.std()  * 100), 2),
            "min_pct":    round(float(arr.min()  * 100), 2),
            "max_pct":    round(float(arr.max()  * 100), 2),
            "per_seed":   [round(v * 100, 2) for v in seed_results],
        }
        results[total_n] = summary
        print(f"  → {summary['mean_pct']:.1f}% ± {summary['std_pct']:.1f}%  "
              f"[{summary['min_pct']:.1f}–{summary['max_pct']:.1f}%]")

    return results


# ── Model B baseline (LMSYS, full 6000 samples, 20 seeds) ────────────────────

def model_b_baseline(lmsys_path: pathlib.Path, n_seeds: int) -> dict:
    print(f"\n{'─'*60}")
    print("  Model B baseline (LMSYS, 6000 samples, held-out 20%)")
    print(f"{'─'*60}")
    df = pd.read_csv(lmsys_path)
    seed_results = []
    for seed in range(n_seeds):
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y
        )
        model = XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X_train, y_train)
        ra = ranking_accuracy(model, X_test, y_test)
        seed_results.append(ra)

    arr = np.array(seed_results)
    summary = {
        "mean_pct": round(float(arr.mean() * 100), 2),
        "std_pct":  round(float(arr.std()  * 100), 2),
    }
    print(f"  → {summary['mean_pct']:.1f}% ± {summary['std_pct']:.1f}%")
    return summary


# ── Output formatting ─────────────────────────────────────────────────────────

def print_summary_table(ablation: dict, baseline: dict) -> None:
    print(f"\n{'═'*65}")
    print("  DOMAIN RETRAINING ABLATION — SUMMARY TABLE")
    print(f"{'═'*65}")
    print(f"  {'N samples':>12}  {'Ranking Acc':>12}  {'± std':>8}  {'Range':>16}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*16}")
    for n, r in sorted(ablation.items()):
        print(f"  {n:>12,}  {r['mean_pct']:>11.1f}%  "
              f"{r['std_pct']:>7.1f}%  "
              f"[{r['min_pct']:.1f}–{r['max_pct']:.1f}%]")
    print(f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*16}")
    print(f"  {'Model B (LMSYS)':>12}  {baseline['mean_pct']:>11.1f}%  "
          f"{baseline['std_pct']:>7.1f}%  {'(baseline)':>16}")
    print(f"{'═'*65}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Domain retraining sample-size ablation")
    parser.add_argument("--domain", default="data/model_d_training_data.csv",
                        help="Featurized serving logs CSV")
    parser.add_argument("--lmsys",  default="data/lmsys_labeled.csv",
                        help="LMSYS training data for Model B baseline")
    parser.add_argument("--sizes",  nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seeds",  type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out",    default="results/domain_retrain_ablation.json")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).parent.parent
    domain_path = repo_root / args.domain
    lmsys_path  = repo_root / args.lmsys
    out_path    = repo_root / args.out

    domain_df = load_serving_logs(domain_path)
    baseline  = model_b_baseline(lmsys_path, args.seeds)
    ablation  = run_ablation(domain_df, lmsys_path, args.sizes, args.seeds)

    print_summary_table(ablation, baseline)

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline_model_b": baseline, "ablation": ablation}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
