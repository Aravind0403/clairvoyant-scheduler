# Clairvoyant Scheduler — Experiment Results Summary Index

This document provides a consolidated, easily accessible record of all empirical benchmark results, feature ablations, baseline comparisons, and sensitivity sweeps for **Clairvoyant**.

---

## 1. Baseline Pairwise Ranking Accuracy (`tab:baselines` / Table 6)

Evaluated on held-out 20% test splits across 3 conversation datasets. Pairwise ranking accuracy measures $P(\text{score}(\text{Long}) > \text{score}(\text{Short}))$:

| Method | P99 Overhead | ShareGPT | LMSYS | OASST1 | Systems Tradeoff |
| --- | --- | --- | --- | --- | --- |
| **FCFS (random)** | 0 ms | 50.0% | 50.0% | 50.0% | Default queueing order. |
| **Prompt-length rule** | <0.01 ms | 52.4% | 52.3% | 55.8% | Naïve prompt token count threshold (>20). |
| **Keyword heuristic** | <0.01 ms | 36.3% | 4.6% | 18.5% | Code keyword rule. Harmful on LMSYS. |
| **DistilBERT Transformer** | **865.0 ms** | **79.9%** | **92.0%** | **69.5%** | Heavy transformer model; **865ms inline delay**. |
| **Clairvoyant (ONNX XGBoost)** | **0.029 ms** | **74.9%** | **95.1%** | **67.1%** | **Optimal**: Near-identical accuracy at 30,000× lower latency! |

---

## 2. Drop-One Feature Ablation Study (`tab:ablation` / Section 6)

Empirical impact on ranking accuracy when removing each feature group from the 19-feature model (trained on 6,000 samples per dataset, seed 42):

| Feature Dropped | ShareGPT ($\Delta$) | LMSYS ($\Delta$) | OASST1 ($\Delta$) | Avg $\Delta$ Rank Acc | Effect |
| --- | --- | --- | --- | --- | --- |
| **`prompt_token_len`** | -3.57 pp | -2.49 pp | -3.21 pp | **-3.09 pp** | Universal primary signal |
| **`instruction_verb` (all 13 verb_*)** | -3.52 pp | -5.04 pp | +3.21 pp | **-1.78 pp** | High impact on LMSYS/conversations |
| **`has_code_keyword`** | -4.47 pp | -0.54 pp | +0.49 pp | **-1.51 pp** | Essential for code tasks (ShareGPT) |
| **`ends_with_question`** | -2.25 pp | -0.30 pp | -0.84 pp | **-1.13 pp** | Useful for Q&A tasks |
| **`has_length_constraint`** | -0.55 pp | -0.21 pp | +0.42 pp | **-0.12 pp** | Neutral signal |
| **`has_format_keyword`** | -0.01 pp | +0.11 pp | +2.24 pp | **+0.78 pp** | Minor redundant signal |
| **`clause_count`** | +0.51 pp | -0.47 pp | +3.18 pp | **+1.07 pp** | Pruning candidate |

---

## 3. Tokenizer Approximation Error Analysis (`len // 4` vs. True Tokens)

Character length rule `len(prompt) // 4` evaluated against exact HuggingFace/tiktoken counts:

| Dataset | Prompts | MAE | Median AE | Pearson Correlation ($r$) | Pairwise Rank Consistency |
| --- | --- | --- | --- | --- | --- |
| **ShareGPT** | 6,000 | 13.36 tok | 4.0 tok | **$r = 0.9300$** | **94.32%** |
| **LMSYS** | 6,000 | 0.00 tok | 0.0 tok | **$r = 1.0000$** | **100.00%** |
| **OASST1** | 828 | 0.00 tok | 0.0 tok | **$r = 1.0000$** | **100.00%** |

---

## 4. Domain Retraining Sample-Size Convergence (`TODO-C`)

Featurized target domain serving logs (`data/model_d_training_data.csv`) evaluated across total balanced sample sizes ($N$) over 5 random seeds:

| N Domain Samples | Per-Class Count | Mean Ranking Acc | Std Dev ($\pm$) | Accuracy Range |
| --- | --- | --- | --- | --- |
| **N = 90** | 30 / class | 79.4% | $\pm$ 13.7% | [58.3% – 100.0%] |
| **N = 180** | 60 / class | 81.2% | $\pm$ 7.0% | [69.4% – 88.2%] |
| **N = 240** | 80 / class | **82.3%** | **$\pm$ 2.9%** | **[77.3% – 85.2%]** |
| **Model B (LMSYS General)** | Baseline | 95.5% | $\pm$ 0.7% | — |

---

## 5. Starvation Timeout ($\tau$) & Workload Spectrum ($\rho$) Sweeps

### A. $\tau$ Sensitivity Sweep ($\rho=0.74, \lambda=0.12$/s)
Discrete-Event Simulation ($n=2,000$ requests, 5 seeds):

| Condition | Short P50 | Short P95 | Long P50 | Long P95 |
| --- | --- | --- | --- | --- |
| **FCFS Baseline** | 9.70s | 43.71s | 15.60s | 51.79s |
| **$\tau = 0.5 \times \mu_{\text{short}}$** | 7.22s | 16.60s | 14.64s | 72.16s |
| **$\tau = 1.0 \times \mu_{\text{short}}$** | 8.38s | 18.15s | 15.18s | 69.35s |
| **$\tau = 2.0 \times \mu_{\text{short}}$** | 9.18s | 20.99s | 16.31s | 64.56s |
| **$\tau = 3.0 \times \mu_{\text{short}}$ (Default)** | **8.03s** | **23.46s** | **16.83s** | **60.45s** |
| **$\tau = 5.0 \times \mu_{\text{short}}$** | 7.02s | 28.56s | 16.07s | 55.17s |
| **$\tau = \infty$ (Pure SJF)** | 5.97s | 14.72s | 14.14s | 79.32s |

### B. Workload Spectrum Sweep ($\tau = 3\times$)

| Utilization ($\rho$) | Arrival Rate $\lambda$ | FCFS P50 | SJF P50 | Reduction | Std Dev |
| --- | --- | --- | --- | --- | --- |
| **0.30** | 0.0484 /s | 3.87s | 3.88s | $-0.4\%$ | $\pm 1.7\%$ |
| **0.50** | 0.0806 /s | 4.53s | 4.41s | $+2.7\%$ | $\pm 3.7\%$ |
| **0.60** | 0.0968 /s | 5.73s | 5.11s | **$+10.9\%$** | $\pm 3.9\%$ |
| **0.74** | 0.1194 /s | 9.58s | 7.92s | **$+17.4\%$** | $\pm 10.1\%$ |
| **0.85** | 0.1371 /s | 15.66s | 14.10s | **$+10.0\%$** | $\pm 14.3\%$ |

---

## 6. End-to-End Latency Benchmarks (RTX 4090, 100 Burst Requests)

| Model | Class | Policy | P50 Latency | P95 Latency | P99 Latency |
| --- | --- | --- | --- | --- | --- |
| **Gemma3:4b** | Short | FCFS | 229.5 s | 504.7 s | 552.2 s |
|  | Short | **SJF** | **69.1 s** (-70%) | **163.0 s** (-68%) | **177.8 s** (-68%) |
|  | Long | FCFS | 309.7 s | 547.0 s | 578.8 s |
|  | Long | **SJF** | 376.0 s | 554.9 s | 573.6 s |
| **Llama3.1:8b** | Short | FCFS | 158.8 s | 352.0 s | 367.7 s |
|  | Short | **SJF** | **38.0 s** (-76%) | **94.8 s** (-73%) | **104.8 s** (-71%) |
|  | Long | FCFS | 188.7 s | 342.2 s | 360.7 s |
|  | Long | **SJF** | 239.6 s | 355.3 s | 367.6 s |

---

## 7. GCP NVIDIA L4 GPU Real Workload Trace Replay Benchmarks

Empirical validation on GCP Compute Engine (`g2-standard-4`, 1x NVIDIA L4 24GB VRAM GPU) replaying real LMSYS conversation prompts under Poisson arrivals ($\rho=0.80$, $E[S]=6.2\text{s}$):

| Policy / Endpoint | Short TTFT P50 | Short Queue Wait P50 | Short End-to-End P50 | Latency Reduction |
| --- | --- | --- | --- | --- |
| **FCFS Baseline (Ollama :11434)** | 10.47 s | 10.47 s | 10.86 s | Baseline |
| **Clairvoyant SJF (Proxy :8080)** | **1.71 s** | **1.71 s** | **1.71 s** | **-83.6% Reduction** ($6.1\times$ faster) |

