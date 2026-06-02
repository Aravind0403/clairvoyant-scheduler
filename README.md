# Clairvoyant

**Backend-agnostic sidecar proxy that eliminates Head-of-Line Blocking in LLM inference via ML-driven Shortest-Job-First scheduling — zero backend modification required.**

![Go](https://img.shields.io/badge/Go-1.21-00ADD8?logo=go) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EA4C1D) ![License](https://img.shields.io/badge/license-MIT-green)

---

## The Problem

LLM inference backends like Ollama and llama.cpp process requests sequentially — one request holds the GPU until generation completes. When a long code-generation job arrives before a simple factual query, the short request waits the full generation time. This is **Head-of-Line Blocking (HOLB)** at Layer 1 (request admission), and it directly inflates P50 latency for every short request in the queue.

```
FCFS (default):    [long ████████████████████] [short ██] [short ██] [short ██]
                    ^ short requests wait the full long generation time

SJF (Clairvoyant): [short ██] [short ██] [short ██] [long ████████████████████]
                    ^ short requests served first; 70–76% P50 reduction on RTX 4090
```

> **Scope:** Clairvoyant targets serial-dispatch backends (Ollama, llama.cpp) — the dominant deployment model for on-premise and edge LLM serving. Continuous-batching engines (vLLM, Orca, TGI) solve a different problem at a different layer and are explicitly out of scope.

---

## How It Works

A **Go HTTP sidecar proxy** intercepts every `/v1/chat/completions` request and:

1. Extracts **19 lexical features** from the prompt — no model call, no tokeniser at runtime
2. Runs an ONNX-exported XGBoost classifier in **0.029 ms** → produces a continuous P(Long) score
3. Pushes the request into a **min-heap priority queue** keyed on ascending P(Long)
4. Dispatches shortest-predicted first, with a calibrated **starvation timeout** τ = 3 × µ_short to prevent indefinite long-job blocking

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
│     ONNX Predictor      │  19 features → XGBoost → P(Short / Medium / Long)
│     0.029 ms/request    │  pure string scan, no tokeniser dependency
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
│  Any OpenAI-compatible  │  backend receives requests in SJF order
│  backend (Ollama, etc.) │  streams response back to original caller
└─────────────────────────┘
```

---

## Results

### GPU Benchmark (RTX 4090, Ollama, n=250 per cell)

End-to-end latency under a 100-request concurrent burst (50 Short + 50 Long, all queued simultaneously).

| Model | Class | Condition | P50 | P95 | P99 |
|-------|-------|-----------|-----|-----|-----|
| Gemma3:4b | SHORT | FCFS | 229.5 s | 504.7 s | 552.2 s |
| Gemma3:4b | SHORT | **SJF** | **69.1 s** | **163.0 s** | **177.8 s** |
| Gemma3:4b | LONG | FCFS | 309.7 s | 547.0 s | 578.8 s |
| Gemma3:4b | LONG | SJF | 376.0 s | 554.9 s | 573.6 s |
| Llama3.1:8b | SHORT | FCFS | 158.8 s | 352.0 s | 367.7 s |
| Llama3.1:8b | SHORT | **SJF** | **38.0 s** | **94.8 s** | **104.8 s** |
| Llama3.1:8b | LONG | FCFS | 188.7 s | 342.2 s | 360.7 s |
| Llama3.1:8b | LONG | SJF | 239.6 s | 355.3 s | 367.6 s |

**70% P50 reduction** for Gemma3:4b short requests; **76%** for Llama3.1:8b. Long-request latency increase is intentional and bounded by τ.

### Workload Spectrum (Poisson Arrivals, DES Simulation)

Under realistic steady-state Poisson arrivals calibrated to RTX 4090 service times (µ_short = 3.5 s, µ_long = 8.9 s, τ = 3 × µ_short = 10.5 s):

| Queue utilisation ρ | SJF P50 reduction | Notes |
|---|---|---|
| 0.30 | ~0% | Arrivals sparse; no queuing |
| 0.50 | ~3% | Negligible |
| 0.60 | ~11% | Meaningful |
| **0.74** | **17% ± 10%** | **Peak benefit** |
| 0.85 | 10% ± 14% | Starvation timeout fires more often |

Practical deployment range: **0.55 ≲ ρ ≲ 0.80**. Below ρ = 0.50, FCFS suffices.

### ML Predictor Accuracy

| Model | Training Dataset | Ranking Acc | Class Acc | Gap |
|-------|-----------------|-------------|-----------|-----|
| A | ShareGPT | **76.3%** | 47.6% | +28.7 pp |
| B | LMSYS-Chat-1M | **95.6%** | 66.8% | +28.8 pp |
| C | OASST1 | **62.2%** | 41.0% | +21.2 pp |

Cross-distribution accuracy (train on one dataset, test on another): **52–66%**. Random baseline: 50%.

### Baseline Comparison

Pairwise ranking accuracy (Short vs. Long pairs) across three evaluation datasets:

| Method | ShareGPT | LMSYS | OASST1 |
|--------|----------|-------|--------|
| FCFS (random) | 50.0% | 50.0% | 50.0% |
| Prompt-length rule | 52.4% | 52.3% | 55.8% |
| Keyword heuristic | 36.3% | 4.6% | 18.5% |
| **Clairvoyant (XGBoost)** | **74.9%** | **95.1%** | **67.1%** |

The keyword heuristic is actively harmful on LMSYS (4.6%), where code vocabulary does not predict long outputs. Clairvoyant outperforms the next-best method by 11–43 pp across all three datasets.

### Predictor Latency

| Method | P50 | P99 | vs Clairvoyant |
|--------|-----|-----|----------------|
| Clairvoyant (19 lexical features, ONNX) | **0.029 ms** | — | baseline |
| Embedding-based (all-MiniLM-L6-v2, CPU) | 12.85 ms | 865.73 ms | **443–5,125× slower** |

Embedding-based prediction approaches generation latency at P99 on consumer hardware, making it incompatible with this deployment regime. The lexical approach is not a compromise — it is the only architecture compatible with the latency budget.

### Starvation Timeout (τ) Sensitivity

Simulation at ρ = 0.74, Poisson arrivals, 5 seeds (service times 𝒩(3.5 s, 0.8 s) short, 𝒩(8.9 s, 2.0 s) long):

| τ | SHORT P50 | SHORT P95 | LONG P50 | LONG P95 |
|---|-----------|-----------|----------|----------|
| FCFS (baseline) | 9.70 s | 43.71 s | 15.60 s | 51.79 s |
| 1.0 × µ_short | 8.38 s | 18.15 s | 15.18 s | 69.35 s |
| **3.0 × µ_short (default)** | **8.03 s** | **23.46 s** | **16.83 s** | **60.45 s** |
| 5.0 × µ_short | 7.02 s | 28.56 s | 16.07 s | 55.17 s |
| ∞ (pure SJF) | 5.97 s | 14.72 s | 14.14 s | 79.32 s |

τ = 3 × µ_short sits at the Pareto elbow: 17% short P50 improvement over FCFS while bounding long P95 penalty to 17%. In the burst benchmark, τ has negligible effect (<1% variation) — all short requests clear within ~175 s regardless of τ.

**Calibration rule:** measure µ_short under representative mixed-workload queueing conditions (not sequential service times). Apple M1 + Ollama → µ_short ≈ 40 s → τ = 120 s. RTX 4090 → µ_short ≈ 3.5 s → τ = 10.5 s.

---

## Why Ranking Accuracy, Not Classification Accuracy

Classification accuracy (41–67%) covers the full three-class problem — Short, Medium, Long. The scheduler only needs one thing: *is this request shorter or longer than that one?* Ranking accuracy measures exactly that: the fraction of (Short, Long) pairs where the model assigns a higher P(Long) to the Long example.

Ranking accuracy (62–96%) exceeds classification accuracy by 21–29 pp across all three models. Medium misclassifications have limited impact on SJF ordering; the Short/Long boundary is what determines whether a request experiences ~2 s or ~30 s of latency.

---

## Dataset Bias Finding

**Curated instruction datasets are systematically unusable as SJF training sources.**

| Dataset | Long (≥800 tokens) | % Long | Usable? |
|---------|-------------------|--------|---------|
| ShareGPT | ~7,800 | ~15% | ✅ |
| LMSYS-Chat-1M | ~120K | ~12% | ✅ |
| OASST1 | 551 | 6.3% | ✅ (limited) |
| Alpaca 52K | 4 | 0.008% | ❌ |
| CodeAlpaca 20K | 3 | 0.015% | ❌ |
| Dolly 15K | 88 | 0.6% | Test-only |
| CNN/DailyMail | 1 | 0.009% | Test-only |

Root cause: GPT-generated instruction datasets impose brevity constraints that eliminate the long-response examples the scheduler must learn to distinguish. Only natural conversation logs provide sufficient Long-class diversity.

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
│   ├── featurize_serving_logs.py  # Model D — production log retraining
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
│   └── feature_extractor.py       # Python reference implementation of features.go
├── results/
│   ├── benchmark_summary_gemma3.csv
│   ├── benchmark_summary_llama.csv
│   ├── workload_spectrum.png      # SJF benefit vs ρ (Figure 3 in paper)
│   ├── workload_spectrum.json     # raw simulation data
│   ├── tau_sensitivity.png
│   └── tau_pareto.png
├── figures/
│   ├── holb_timeline.svg/.pdf     # HOLB timeline diagram (Figure 1)
│   └── architecture.svg/.pdf      # system architecture diagram (Figure 2)
├── scheduler/
│   ├── config/config.go
│   ├── predictor/
│   │   ├── features.go            # Go port of feature extractor
│   │   └── onnx.go                # onnxruntime_go wrapper
│   ├── queue/queue.go             # min-heap + starvation timeout
│   ├── proxy/proxy.go             # HTTP intercept and response streaming
│   └── main.go
└── tests/
    └── test_ordering_dolly.py     # n=8 ordering-correctness test (Dolly 15K prompts)
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
python data/pipeline/featurize_serving_logs.py --input your_logs.jsonl
python model/train.py
python model/export.py   # produces drop-in replacement: model/predictor.onnx
```

At ~500 balanced Short/Long examples, XGBoost training completes in under 10 seconds on a consumer CPU.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | Proxy listen address |
| `BACKEND_URL` | `http://localhost:11434` | Upstream inference backend |
| `QUEUE_CAPACITY` | `256` | Max queued requests (429 on overflow) |
| `STARVATION_TIMEOUT_SEC` | `15` | τ in seconds — set to 3 × µ_short under queueing conditions |
| `ONNX_MODEL_PATH` | `model/predictor.onnx` | Path to ONNX model |
| `ONNX_LIB_PATH` | *(system linker)* | Path to `libonnxruntime.so/.dylib` |

---

## Known Issue: XGBoost 2.x + ONNX Export

XGBoost 2.x omits `split_condition` from `get_dump(format='json')` for binary one-hot features, causing `onnxmltools` to fail silently on incomplete internal nodes.

**Fix in `model/export.py`:**

```python
def fix_split_conditions(dump):
    for node in dump:
        if 'split_condition' not in node and 'children' in node:
            node['split_condition'] = 1.0  # correct threshold for binary 0/1 features
    return dump
# Apply before onnxmltools.convert.convert_xgboost(), then remove patched nodes
```

This affects any XGBoost 2.x model with binary features and will fail silently without the patch.

---

## Paper

**Clairvoyant: Predictive SJF Scheduling for Head-of-Line Blocking Mitigation in LLM Serving**  
Aravind Sundaresan — Independent Researcher — aravindsharma20@gmail.com  
arXiv preprint — https://github.com/Aravind0403/clairvoyant-scheduler

---

## License

MIT
