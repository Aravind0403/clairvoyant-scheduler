# Clairvoyant

**Backend-agnostic sidecar proxy that eliminates Head-of-Line Blocking in LLM inference via ML-driven Shortest-Job-First scheduling — zero backend modification required.**

![Go](https://img.shields.io/badge/Go-1.21-00ADD8?logo=go) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EA4C1D) ![Status](https://img.shields.io/badge/Paper-In%20Preparation-orange)

---

## The Problem

LLM inference backends like Ollama process requests sequentially — one request holds the GPU until generation completes. When a long code-generation job arrives before a simple one-sentence query, the short request waits. This is **Head-of-Line Blocking (HOLB)** at Layer 1 (request admission), and it directly inflates P50 latency for every short request in the queue.

```
FCFS (default):    [long ████████████████████] [short ██] [short ██] [short ██]
                    ^ short requests wait the full long generation time

SJF (Clairvoyant): [short ██] [short ██] [short ██] [long ████████████████████]
                    ^ short requests served first — 44% lower P50
```

> **Scope:** Clairvoyant targets serial-dispatch backends (Ollama, single-process model servers) — the dominant deployment model for on-premise LLM serving at small-to-medium scale. vLLM's continuous batching solves a different layer of the problem (within-batch scheduling, not admission ordering) and is explicitly out of scope.

---

## How It Works

A **Go HTTP sidecar proxy** intercepts every `/v1/chat/completions` request and:

1. Extracts **19 lexical features** from the prompt — no model call, no tokeniser at runtime
2. Runs an ONNX-exported XGBoost classifier in **<0.029ms** → predicts Short / Medium / Long output
3. Pushes the request into a **min-heap priority queue** keyed on predicted class + enqueue time
4. Dispatches shortest-predicted first, with a configurable **starvation timeout** (τ) to prevent long requests from waiting indefinitely

```
Incoming Request
      │
      ▼
┌─────────────────────────┐
│     Go HTTP Proxy       │  intercepts /v1/chat/completions
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     ONNX Predictor      │  19 features → XGBoost → Short / Medium / Long
│     <0.029ms            │  len//4 approximation at runtime (no tokeniser dep)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     SJF Priority Queue  │  min-heap: (predicted_class, enqueue_time)
│     + Aging Monitor     │  Long requests promoted after τ = 3× short latency
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     Ollama / any        │  backend receives requests in SJF order
│     OAI-compatible API  │  streams response back to original caller
└─────────────────────────┘
```

---

## Results

| Metric | Value |
|--------|-------|
| ONNX inference latency | **0.029ms** (344× under the 10ms target) |
| Ranking accuracy | **76.29%** (random baseline: 33%) |
| SJF ordering accuracy | **100%** at τ = 120s (Apple M1, Gemma3:4b) |
| P50 latency reduction — short requests | **44%** vs FCFS |
| Backend modification required | **None** |

**Starvation timeout formula** — empirically derived, hardware-agnostic:

```
τ = 3 × expected_short_request_latency
```

Validated across Apple M1 Pro (τ=120s, ~40s/request) and NVIDIA RTX 4090 (τ=15s, ~4s/request).

GPU benchmark across two model families (Gemma3:4b, Llama3.1:8b) with P50/P95/P99 measurements in progress.

---

## Why Ranking Accuracy, Not Classification Accuracy

Classification accuracy is **47.6%** (random baseline: 33.3%). Ranking accuracy is **76.29%**. The scheduler needs to order Short before Long — not predict exact token counts. Medium misclassification has limited impact on SJF ordering. Ranking accuracy captures the metric that actually matters.

The core finding: **19 surface-level lexical features are sufficient for ranking quality that produces correct SJF ordering**. Code requests are longer than factual questions regardless of which dataset or model you train on — because `has_code_keyword` fires on vocabulary, not model behaviour.

---

## Repository Structure

```
clairvoyant/
├── data/pipeline/
│   ├── download.py       # pull ShareGPT from HuggingFace
│   ├── clean.py          # first-turn extraction, length filters
│   └── featurize.py      # 19-feature extraction, BERT tokeniser, labelling
├── model/
│   ├── train.py          # XGBoost, 5-fold CV, 3-class softmax
│   ├── export.py         # ONNX export + XGBoost 2.x bug fix
│   ├── evaluate.py       # ranking accuracy + classification accuracy
│   └── predictor.onnx    # exported model
├── profiler/
│   ├── profiler.py       # end-to-end profiling harness
│   └── feature_extractor.py  # Python reference (mirrors features.go)
└── scheduler/
    ├── config/config.go      # env-var configuration
    ├── predictor/
    │   ├── features.go       # Go port of feature extractor
    │   └── onnx.go           # onnxruntime_go wrapper
    ├── queue/queue.go        # min-heap + aging monitor
    ├── proxy/proxy.go        # HTTP intercept and response streaming
    ├── main.go
    └── test_queue_ordering.py  # integration test harness
```

---

## Quick Start

**Prerequisites:** Go 1.21+, Python 3.10+, `libonnxruntime` installed, Ollama running locally.

```bash
# Install libonnxruntime (macOS)
brew install onnxruntime

# Build
cd scheduler
go build -o clairvoyant ./cmd/main.go

# Run (defaults: :8080 → localhost:11434)
ONNX_MODEL_PATH=../model/predictor.onnx \
ONNX_LIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
STARVATION_TIMEOUT_SEC=120 \
./clairvoyant
```

Point your client at `http://localhost:8080` instead of `http://localhost:11434`. No other changes.

> **macOS note:** Go's HTTP server binds IPv6 (`::1`) by default. Use `http://[::1]:8080` in test scripts, not `http://localhost:8080`.

### Retrain on your own serving logs

The architecture separates the scheduler (fixed, deploy once) from the predictor (workload-specific). To retrain on your deployment's actual traffic:

```bash
# Collect prompts + actual output token counts from your backend
# then retrain with the same 19 features — no scheduler changes needed
python data/pipeline/featurize.py --input your_serving_logs.jsonl
python model/train.py
python model/export.py   # drop-in replacement: model/predictor.onnx
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | Proxy listen address |
| `BACKEND_URL` | `http://localhost:11434` | Upstream inference backend |
| `QUEUE_CAPACITY` | `256` | Max queued requests (429 on overflow) |
| `STARVATION_TIMEOUT_SEC` | `15` | Seconds before Long request gets priority boost |
| `ONNX_MODEL_PATH` | `model/predictor.onnx` | Path to ONNX model |
| `ONNX_LIB_PATH` | *(system linker)* | Path to `libonnxruntime.so/.dylib` |

**Starvation timeout rule of thumb:** `τ = 3 × expected_short_request_latency`. GPU (RTX 4090, ~4s/req) → 15s. Apple M1 + Ollama (~40s/req) → 120s.

---

## Known Issue: XGBoost 2.x + ONNX Export

XGBoost 2.x omits `split_condition` from `get_dump(format='json')` for binary one-hot features. `onnxmltools` fails silently on incomplete internal nodes.

**Fix in `model/export.py`:**

```python
def fix_split_conditions(dump):
    for node in dump:
        if 'split_condition' not in node and 'children' in node:
            node['split_condition'] = 1.0  # correct threshold for binary 0/1 features
    return dump
# Apply before onnxmltools.convert.convert_xgboost(), remove after
```

This is a production hazard — any XGBoost 2.x model with binary features will hit this without warning.

---

## Paper

**Clairvoyant: Eliminating Head-of-Line Blocking in LLM Inference via Lexical-Feature SJF Scheduling**

In preparation. Targeting MLSys 2026.

---

## License

MIT
