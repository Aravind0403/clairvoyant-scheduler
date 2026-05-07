"""
Clairvoyant Scheduler — Export predictor_v3.json → predictor.onnx

Converter: onnxmltools (standard XGBoost → ONNX path).

XGBoost 2.x compatibility note
-------------------------------
XGBoost 2.x omits `split_condition` in get_dump(format='json') for binary
(0/1) one-hot features — the value IS stored in the model (always 1.0) but
the dump skips it. Both onnxmltools and hummingbird-ml read that dump text
and crash on the missing key. We monkey-patch get_dump to reinject it before
the converter ever sees the output. Feature names are also remapped to f0..fN
since both converters parse them as integer indices.

Steps
-----
1. Rebuild the feature matrix from training_data.csv
2. Load predictor_v3.json with XGBoost
3. Patch booster: remap feature names + fix binary split nodes in dump
4. Convert to ONNX via onnxmltools, restore booster state
5. Save to model/predictor.onnx
6. Verify with onnxruntime: 5 sample predictions + latency

Usage
-----
  python model/export.py [--data  data/training_data.csv]
                         [--model model/predictor_v3.json]
                         [--out   model/predictor.onnx]
"""

import argparse
import json as _json
import pathlib
import time

import numpy as np
import pandas as pd
import onnxruntime as rt
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.utils import save_model as onnx_save
from onnxmltools.convert.common.data_types import FloatTensorType


# ─────────────────────────────────────────────────────────────────────────────
# Feature schema — must match train.py exactly
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "Short", 1: "Medium", 2: "Long"}

NUMERIC_COLS = [
    "prompt_token_len", "has_code_keyword", "has_length_constraint",
    "ends_with_question", "has_format_keyword", "clause_count",
]

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define", "other",
]


def build_features(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["verb_norm"] = df["instruction_verb"].where(
        df["instruction_verb"].isin(KNOWN_VERBS), other="other"
    )
    verb_dummies = pd.get_dummies(df["verb_norm"], prefix="verb").reindex(
        columns=[f"verb_{v}" for v in KNOWN_VERBS], fill_value=0
    )
    present = [c for c in NUMERIC_COLS if c in df.columns]
    return pd.concat([df[present].astype(float), verb_dummies], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost 2.x dump patcher
# ─────────────────────────────────────────────────────────────────────────────

def _patch_node(node: dict) -> None:
    """Recursively add split_condition=1.0 to any internal node missing it.

    XGBoost 2.x omits split_condition for binary one-hot features (0/1) in
    the JSON dump.  The true threshold is always 1.0 (feature < 1.0 → left).
    We also set missing → yes so the converter has a valid missing direction.
    """
    if "children" in node:
        if "split_condition" not in node:
            node["split_condition"] = 1.0
            node.setdefault("missing", node["yes"])
        for child in node["children"]:
            _patch_node(child)


def _make_patched_get_dump(original_get_dump):
    """Return a wrapper that fixes the JSON dump before returning it."""
    def patched(*args, **kwargs):
        dumps = original_get_dump(*args, **kwargs)
        fmt = kwargs.get("dump_format", args[2] if len(args) > 2 else "text")
        if fmt == "json":
            result = []
            for d in dumps:
                tree = _json.loads(d)
                _patch_node(tree)
                result.append(_json.dumps(tree))
            return result
        return dumps
    return patched


def _section(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_path:  pathlib.Path,
    model_path: pathlib.Path,
    out_path:   pathlib.Path,
) -> None:

    # ── 1. feature matrix ────────────────────────────────────────────────────
    print(f"Building features from {data_path} …")
    X = build_features(data_path)
    X_np = X.values.astype(np.float32)
    n_features = X_np.shape[1]
    print(f"  Shape: {X_np.shape}   dtype: {X_np.dtype}")

    # ── 2. load XGBoost ───────────────────────────────────────────────────────
    _section("Loading XGBoost model")
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(model_path))
    xgb_kb = model_path.stat().st_size / 1024
    print(f"  Loaded: {model_path}  ({xgb_kb:.1f} KB)")

    # ── 3. patch booster for onnxmltools ─────────────────────────────────────
    _section("Patching booster for ONNX conversion")
    booster = xgb_model.get_booster()
    original_names = booster.feature_names

    # remap feature names → f0..fN (converters parse them as integer indices)
    booster.feature_names = [f"f{i}" for i in range(n_features)]

    # inject missing split_condition for XGBoost 2.x binary split nodes
    original_get_dump = booster.get_dump
    booster.get_dump = _make_patched_get_dump(original_get_dump)

    print("  ✓ Feature names remapped to f0..fN")
    print("  ✓ get_dump patched to add split_condition for binary nodes")

    # ── 4. convert to ONNX ───────────────────────────────────────────────────
    _section("Converting to ONNX  (onnxmltools)")
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_types)

    # restore booster to original state
    booster.feature_names = original_names
    booster.get_dump = original_get_dump

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_save(onnx_model, str(out_path))
    onnx_kb = out_path.stat().st_size / 1024
    print(f"  Saved: {out_path}")
    print(f"  Size: XGBoost JSON {xgb_kb:.1f} KB → ONNX {onnx_kb:.1f} KB")

    # ── 5. verify with onnxruntime ────────────────────────────────────────────
    _section("Verifying with onnxruntime")
    sess = rt.InferenceSession(str(out_path))

    input_name  = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    out_names   = [o.name for o in sess.get_outputs()]
    out_shapes  = [o.shape for o in sess.get_outputs()]

    print(f"  Input   name={input_name!r}   shape={input_shape}")
    for name, shape in zip(out_names, out_shapes):
        print(f"  Output  name={name!r}   shape={shape}")

    # ── 6. sample predictions ────────────────────────────────────────────────
    _section("Sample predictions  (5 rows)")
    sample_idx = [0, 1000, 2000, 3000, 4999]
    X_sample   = X_np[sample_idx]

    xgb_preds  = xgb_model.predict(X_sample).astype(int)
    onnx_raw   = sess.run(None, {input_name: X_sample})
    onnx_preds = np.array(onnx_raw[0]).flatten().astype(int)

    # latency: single row averaged over N runs
    N_LATENCY  = 200
    t0 = time.perf_counter()
    for _ in range(N_LATENCY):
        sess.run(None, {input_name: X_sample[:1]})
    latency_ms = (time.perf_counter() - t0) / N_LATENCY * 1000

    header = f"  {'idx':>6}  {'XGB':>8}  {'ONNX':>8}  {'match':>6}  prompt_token_len"
    print(header)
    print("  " + "─" * (len(header) - 2))
    all_match = True
    for i, row_idx in enumerate(sample_idx):
        xp = int(xgb_preds[i]);  op = int(onnx_preds[i])
        tok = int(X_np[row_idx, 0])
        match = "✓" if xp == op else "✗"
        if xp != op:
            all_match = False
        print(f"  {row_idx:>6}  {LABEL_NAMES[xp]:>8}  {LABEL_NAMES[op]:>8}  {match:>6}  {tok}")

    # ── report ────────────────────────────────────────────────────────────────
    _section("Export Report")
    print(f"  XGBoost JSON size       : {xgb_kb:.1f} KB")
    print(f"  ONNX file size          : {onnx_kb:.1f} KB")
    print(f"  Input shape             : {input_shape}")
    print(f"  Output shapes           : {out_shapes}")
    print(f"  Predictions match (5)   : {'✓ all match' if all_match else '✗ mismatch — see above'}")
    print(f"  ONNX latency (1 row)    : {latency_ms:.3f} ms  (avg over {N_LATENCY} runs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export XGBoost → ONNX")
    parser.add_argument("--data",  default="data/training_data.csv")
    parser.add_argument("--model", default="model/predictor_v3.json")
    parser.add_argument("--out",   default="model/predictor.onnx")
    args = parser.parse_args()

    main(
        data_path=pathlib.Path(args.data),
        model_path=pathlib.Path(args.model),
        out_path=pathlib.Path(args.out),
    )
