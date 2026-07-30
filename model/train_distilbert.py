"""
train_distilbert.py — DistilBERT ranking-accuracy baseline for Table 6 (TODO-B)

Trains DistilBERT-base-uncased on each dataset (ShareGPT, LMSYS, OASST1) and
evaluates pairwise ranking accuracy (Short vs Long pairs) — same metric and
same 80/20 split as baseline_table.py and the paper's Table 6.

Run from repo root with conda base (has torch + transformers):
    python model/train_distilbert.py
    python model/train_distilbert.py --datasets ShareGPT OASST1  # skip LMSYS
    python model/train_distilbert.py --epochs 1 --quick          # smoke-test

Requirements (conda base):
    pip install torch transformers scikit-learn pandas
    # MPS (Apple Silicon) used automatically if available

Output:
    results/distilbert_baseline.json   — full results
    Prints table rows ready to paste into tab:baselines
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    print("ERROR: torch / transformers not found.")
    print("Install: pip install torch transformers")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

LABEL_BINS  = [0, 200, 800, float("inf")]
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 128      # prompt tokens — matches inference constraint
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 2e-5

DATASETS = {
    "ShareGPT": "data/training_data.csv",
    "LMSYS":    "data/lmsys_labeled.csv",
    "OASST1":   "data/oasst1_labeled.csv",
}

ROOT = pathlib.Path(__file__).parent.parent


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], labels: list[int],
                 tokenizer: DistilBertTokenizerFast):
        self.encodings = tokenizer(
            prompts, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Ranking accuracy ──────────────────────────────────────────────────────────

def ranking_accuracy_from_probs(p_long: np.ndarray, y: np.ndarray) -> float:
    """P(correct ordering) over all (Short, Long) pairs."""
    idx_short = np.where(y == 0)[0]
    idx_long  = np.where(y == 1)[0]
    if not len(idx_short) or not len(idx_long):
        return float("nan")
    correct = (p_long[idx_long][:, None] > p_long[idx_short][None, :]).sum()
    return float(correct) / (len(idx_long) * len(idx_short))


# ── Train + evaluate one dataset ─────────────────────────────────────────────

def run_dataset(name: str, csv_path: pathlib.Path,
                tokenizer: DistilBertTokenizerFast,
                device: torch.device,
                epochs: int, quick: bool) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Dataset: {name}")
    print(f"{'═'*60}")

    df = pd.read_csv(csv_path)
    df = df[df["prompt"].notna() & (df["actual_output_tokens"] > 0)].copy()

    # Binary label: Short=0, Long=1 (drop Medium for ranking eval)
    df["label_3"] = pd.cut(
        df["actual_output_tokens"],
        bins=LABEL_BINS, labels=[0, 1, 2], right=False,
    ).astype(int)
    df_sl = df[df["label_3"] != 1].copy()   # Short + Long only
    df_sl["label"] = (df_sl["label_3"] == 2).astype(int)   # Long=1, Short=0

    print(f"  Short: {(df_sl['label']==0).sum():,}  "
          f"Long: {(df_sl['label']==1).sum():,}  "
          f"(Medium excluded from ranking eval)")

    if quick:
        df_sl = df_sl.sample(n=min(400, len(df_sl)), random_state=42)
        print(f"  Quick mode: using {len(df_sl)} samples")

    X_train, X_test = train_test_split(
        df_sl, test_size=0.20, random_state=42, stratify=df_sl["label"]
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    train_ds = PromptDataset(
        X_train["prompt"].tolist(), X_train["label"].tolist(), tokenizer
    )
    test_ds = PromptDataset(
        X_test["prompt"].tolist(), X_test["label"].tolist(), tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  ({elapsed:.0f}s)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())

    p_long = np.array(all_probs)
    y      = np.array(all_labels)
    ra     = ranking_accuracy_from_probs(p_long, y)

    # Classification accuracy for reference
    preds = (p_long >= 0.5).astype(int)
    cls_acc = (preds == y).mean()

    print(f"\n  Ranking accuracy : {ra*100:.1f}%")
    print(f"  Classification   : {cls_acc*100:.1f}%")
    print(f"  Test pairs       : {(y==0).sum()} Short × {(y==1).sum()} Long "
          f"= {(y==0).sum()*(y==1).sum():,} pairs")

    return {
        "dataset":             name,
        "ranking_accuracy_pct": round(ra * 100, 1),
        "classification_acc_pct": round(cls_acc * 100, 1),
        "n_train":             len(X_train),
        "n_test":              len(X_test),
        "n_short_test":        int((y == 0).sum()),
        "n_long_test":         int((y == 1).sum()),
        "epochs":              epochs,
        "model":               MODEL_NAME,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DistilBERT ranking baseline (TODO-B)")
    parser.add_argument("--datasets", nargs="+",
                        default=["ShareGPT", "LMSYS", "OASST1"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick",  action="store_true",
                        help="400-sample smoke-test per dataset (~5 min total)")
    parser.add_argument("--out", default="results/distilbert_baseline.json")
    args = parser.parse_args()

    device    = get_device()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f"Tokenizer loaded: {MODEL_NAME}  max_len={MAX_LEN}")

    results = []
    for name in args.datasets:
        path = ROOT / DATASETS[name]
        if not path.exists():
            print(f"\n[SKIP] {name} — not found: {path}")
            continue
        r = run_dataset(name, path, tokenizer, device, args.epochs, args.quick)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print("  DISTILBERT BASELINE — RANKING ACCURACY (paste into tab:baselines)")
    print(f"{'═'*65}")
    row = {r["dataset"]: r["ranking_accuracy_pct"] for r in results}
    for ds in ["ShareGPT", "LMSYS", "OASST1"]:
        val = f"{row[ds]:.1f}%" if ds in row else "---"
        print(f"  DistilBERT  {ds:<12} {val}")

    print(f"\n  Note: P99 inference latency = 865ms (from §3.3) applies across all datasets.")
    print(f"  Fill tab:baselines 'DistilBERT' row with these numbers.")

    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
