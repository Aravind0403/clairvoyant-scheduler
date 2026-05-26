# Clairvoyant

**Backend-agnostic sidecar proxy that eliminates Head-of-Line Blocking in LLM inference via ML-driven Shortest-Job-First scheduling — zero backend modification required.**

![Go](https://img.shields.io/badge/Go-1.21-00ADD8?logo=go) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EA4C1D) ![Status](https://img.shields.io/badge/arXiv-Submission%20Pending-orange)

---

## The Problem

LLM inference backends like Ollama process requests sequentially — one request holds the GPU until generation completes. When a long code-generation job arrives before a simple one-sentence query, the short request waits. This is **Head-of-Line Blocking (HOLB)** at Layer 1 (request admission), and it directly inflates P50 latency for every short request in the queue.

```
FCFS (default):    [long ████████████████████] [short ██] [short ██] [short ██]
                    ^ short requests wait the full long generation time

SJF (Clairvoyant): [short ██] [short ██] [short ██] [long ████████████████████]
                    ^ short requests served first; 70–76% P50 reduction on RTX 4090
```

> **Scope:** Clairvoyant targets serial-dispatch backends (Ollama, llama.cpp in default configuration) — the dominant deployment model for on-premise LLM serving at small-to-medium scale. vLLM's continuous batching solves a different layer of the problem (within-batch scheduling, not admission ordering) and is explicitly out of scope.

---

## How It Works

A **Go HTTP sidecar proxy** intercepts every `/api/generate` request and:

1. Extracts **19 lexical features** from the prompt — no model call, no tokeniser at runtime
2. Runs an ONNX-exported XGBoost classifier in **0.029ms** → produces a continuous P(Long) score
3. Pushes the request into a **min-heap priority queue** keyed on ascending P(Long)
4. Dispatches shortest-predicted first, with a configurable **starvation timeout** (τ) to prevent long requests from waiting indefinitely

```
Incoming Request
      │
      ▼
┌─────────────────────────┐
│     Go HTTP Proxy       │  intercepts /api/generate
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     ONNX Predictor      │  19 features → XGBoost → P(Short / Medium / Long)
│     0.029ms/request     │  pure string scan, no tokeniser dependency
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     SJF Priority Queue  │  min-heap keyed on P(Long) ascending
│     + Starvation Guard  │  requests promoted after τ = 3 × µ_short
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

### GPU Benchmark (RTX 4090, Ollama, n=250 per cell)

End-to-end latency under a 100-request concurrent burst (50 short + 50 long, all queued simultaneously). Five independent runs per condition.

| Model | Class | FCFS P50 | SJF P50 | Reduction |
|-------|-------|----------|---------|-----------|
| Gemma3:4b | SHORT | 229.5s | 69.1s | **70%** |
| Gemma3:4b | LONG | 309.7s | 376.0s | −21% (expected) |
| Llama3.1:8b | SHORT | 158.8s | 38.0s | **76%** |
| Llama3.1:8b | LONG | 188.7s | 239.6s | −27% (expected) |

P95 and P99 reductions are consistent at 68–73% for short requests across both model families. Long-request latency increase is expected and bounded by the starvation timeout (τ = 15s used; recommendation is τ = 3 × µ_short = 10.5s on this hardware).

### Workload Spectrum (Poisson Arrivals, DES Simulation)

The burst benchmark is a worst-case upper bound. Under realistic steady-state Poisson arrivals calibrated to RTX 4090 service times (µ_short = 3.5s, µ_long = 8.9s):

| Queue utilisation ρ | SJF P50 reduction | Notes |
|---|---|---|
| 0.30 | ~0% | Arrivals sparse; no queuing |
| 0.50 | ~3% | Negligible |
| 0.60 | ~11% | Meaningful |
| **0.74** | **17% ± 10%** | **Peak benefit — recommended operating range** |
| 0.85 | 10% ± 14% | Starvation timeout fires more often; variance rises |

Practical deployment range: **0.55 ≲ ρ ≲ 0.80**. Estimate your ρ = λ · E[S] from request logs.

### ML Predictor Accuracy

| Model | Training Dataset | In-dist Ranking Acc | In-dist Class Acc | Gap |
|-------|-----------------|--------------------|--------------------|-----|
| A | ShareGPT | **76.3%** | 47.6% | +28.7pp |
| B | LMSYS-Chat-1M | **95.6%** | 66.8% | +28.8pp |
| C | OASST1 | **62.2%** | 41.0% | +21.2pp |

Cross-distribution accuracy (training on one dataset, tested on another): **52–66%**. Random baseline: 50%.

Ranking accuracy consistently exceeds classification accuracy by 21–29pp because the scheduler needs correct *pairwise ordering* (Short before Long), not exact class labels.

### Predictor Latency

| Method | P50 | P99 | vs Clairvoyant |
|--------|-----|-----|----------------|
| Clairvoyant (19 lexical features, ONNX) | **0.029ms** | — | baseline |
| Embedding-based (all-MiniLM-L6-v2, CPU) | 12.85ms | 865.73ms | **443–5,125× slower** |

Embedding-based prediction is not viable for this deployment target. The lexical approach is not a compromise — it is the only architecture compatible with the latency requirements.

### Starvation Timeout (τ) Sensitivity

Simulation at ρ = 0.74, Poisson arrivals, 5 seeds:

| τ | SHORT P50 | SHORT P95 | LONG P95 |
|---|-----------|-----------|----------|
| FCFS (baseline) | 9.70s | 43.71s | 51.79s |
| 1.0 × µ_short | 8.38s | 18.15s | 69.35s |
| **3.0 × µ_short (default)** | **8.03s** | **23.46s** | **60.45s** |
| 5.0 × µ_short | 7.02s | 28.56s | 55.17s |
| ∞ (pure SJF) | 5.97s | 14.72s | 79.32s |

τ = 3 × µ_short is the Pareto elbow: 17% short P50 improvement while holding long P95 penalty proportionate.

**τ rule of thumb:** Apple M1 + Ollama (~40s/req) → τ = 120s. RTX 4090 (~3.5s/req) → τ = 10.5s.

---

## Why Ranking Accuracy, Not Classification Accuracy

Classification accuracy (41–67%) includes the full three-class problem — short, medium, long. But the scheduler only cares about one question: *is this request shorter or longer than that one?* Ranking accuracy measures exactly that: the fraction of (Short, Long) pairs where the model assigns a higher P(Long) to the Long example.

Ranking accuracy (62–96%) exceeds classification accuracy by 21–29pp across all three models. Medium misclassifications have limited impact on SJF ordering; the short/long boundary is what determines whether a request waits 4s or 75s.

---

## Dataset Bias Finding

A key empirical result: **curated instruction datasets are systematically unusable as SJF training sources**.

| Dataset | Long examples (≥800 tokens) | % Long | Usable? |
|---------|---------------------------|--------|---------|
| ShareGPT | ~7,800 | ~15% | ✅ |
| LMSYS-Chat-1M | ~120K | ~12% | ✅ |
| OASST1 | 551 | 6.3% | ✅ (limited) |
| Alpaca 52K | 4 | 0.008% | ❌ |
| CodeAlpaca 20K | 3 | 0.015% | ❌ |
| Dolly 15K | 88 | 0.6% | Test-only |

Root cause: GPT-generated instruction datasets impose brevity constraints that eliminate the long-tail generation behaviour the scheduler needs to distinguish. Only natural conversation logs provide sufficient Long-class diversity.

---

## Repository Structure

```
clairvoyant/
├── data/pipeline/
│   ├── download.py                # pull ShareGPT from HuggingFace
│   ├── clean.py                   # first-turn extraction, length filters
│   ├── featurize.py               # 19-feature extraction, labelling
│   ├── collect_lmsys_labeled.py   # LMSYS-Chat-1M, existing response text
│   ├── collect_oasst1.py          # OASST1, parent-child pairing, EN only
│   ├── collect_dolly.py           # Dolly 15K (test-only)
│   ├── collect_cnn_dailymail.py   # CNN/DM RAG surrogate (test-only)
│   ├── featurize_serving_logs.py  # for Model D — production log retraining
│   └── run_serving_logs.py        # orchestration for Model D pipeline
├── model/
│   ├── train.py                   # XGBoost, 3-class softmax, 80/20 stratified split
│   ├── export.py                  # ONNX export + XGBoost 2.x split_condition fix
│   ├── evaluate.py                # ranking accuracy + classification accuracy
│   ├── evaluate_ranking.py        # cross-distribution matrix (--matrix flag)
│   ├── ablation.py                # drop-one feature study
│   ├── baseline_table.py          # prompt-length rule + keyword heuristic baselines
│   └── tau_sensitivity.py         # τ Pareto sweep + ρ workload spectrum simulation
├── profiler/
│   ├── benchmark.py               # RTX 4090 GPU benchmark (asyncio.gather burst)
│   └── feature_extractor.py       # Python reference (mirrors features.go)
├── results/
│   ├── benchmark_summary_gemma3.csv
│   ├── benchmark_summary_llama.csv
│   ├── tau_sensitivity.png
│   ├── tau_pareto.png
│   ├── workload_spectrum.png      # SJF benefit vs ρ (Fig 3 in paper)
│   └── workload_spectrum.json     # raw simulation data
├── figures/
│   ├── holb_timeline.svg/.pdf     # HOLB timeline diagram (Fig 1)
│   └── architecture.svg/.pdf      # system architecture diagram (Fig 2)
├── scheduler/
│   ├── config/config.go
│   ├── predictor/
│   │   ├── features.go            # Go port of feature extractor
│   │   └── onnx.go                # onnxruntime_go wrapper
│   ├── queue/queue.go             # min-heap + starvation timeout
│   ├── proxy/proxy.go             # HTTP intercept and response streaming
│   └── main.go
└── tests/
    └── test_ordering_dolly.py     # n=8 ordering-correctness test (Dolly 15K)
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

Point your client at `http://localhost:8080` instead of `http://localhost:11434`. No other changes required.

> **macOS note:** Go's HTTP server binds IPv6 (`::1`) by default. Use `http://[::1]:8080` in test scripts, not `http://localhost:8080`.

### Retrain on your own serving logs

```bash
# Collect prompts + actual output lengths from your backend, then:
python data/pipeline/featurize_serving_logs.py --input your_logs.jsonl
python model/train.py
python model/export.py   # produces drop-in replacement: model/predictor.onnx
```

At ≥500 balanced Short/Long examples, XGBoost training completes in under 10 seconds.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | Proxy listen address |
| `BACKEND_URL` | `http://localhost:11434` | Upstream inference backend |
| `QUEUE_CAPACITY` | `256` | Max queued requests (429 on overflow) |
| `STARVATION_TIMEOUT_SEC` | `15` | τ in seconds — set to 3 × expected short-request latency |
| `ONNX_MODEL_PATH` | `model/predictor.onnx` | Path to ONNX model |
| `ONNX_LIB_PATH` | *(system linker)* | Path to `libonnxruntime.so/.dylib` |

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
# Apply before onnxmltools.convert.convert_xgboost(), then remove the patched nodes
```

This affects any XGBoost 2.x model with binary features — it will fail silently without this patch.

---

## Paper

**Clairvoyant: Predictive SJF Scheduling for Head-of-Line Blocking Mitigation in LLM Serving**  
Aravind Sundaresan — Independent Researcher  
arXiv preprint (submission pending). Target venue: MLSys 2027.

---

## License

MIT
