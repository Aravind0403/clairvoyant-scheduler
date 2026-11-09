# GEEKOM A9 Mega — Independent Evaluation Protocol

**System:** Clairvoyant Predictive Shortest-Job-First (SJF) Admission Proxy  
**Target Hardware:** GEEKOM A9 Mega (AMD Ryzen AI Max+ 395, 128GB Unified Memory, ROCm)  
**Author:** Aravind Sundaresan — Anavind Labs (<aravind@anavindlabs.com>)  
**Preprint Reference:** arXiv:2606.07248 | Code: [github.com/Aravind0403/clairvoyant-scheduler](https://github.com/Aravind0403/clairvoyant-scheduler)

---

## 1. Executive Objective

Evaluate whether predictive Shortest-Job-First (SJF) request admission eliminates Head-of-Line Blocking (HOLB) for interactive LLM queries on unified memory mini-PC hardware (GEEKOM A9 Mega) running large-scale quantized models (70B-class), and quantify the latency tradeoffs for long generation tasks and overall throughput.

---

## 2. Hardware & Access Scope

* **Node Scope:** 1× standalone GEEKOM A9 Mega unit (a 4-node cluster is explicitly not required; single-node admission isolates scheduling dynamics).
* **Permissions:** Standard user-level shell account with ROCm device access (`/dev/kfd`, `/dev/dri`) and write permissions in a dedicated home workspace. Root access is neither requested nor required.
* **Testing Window:** 
  * Phase 1: 30–45 minute smoke test (environment verification, ROCm check, single-prompt end-to-end baseline, empirical 70B decode speed and $\mathbb{E}[S_{\text{short}}]$ measurement).
  * Phase 2: Dedicated benchmark windows scheduled during off-peak hours (UTC+8) to eliminate latency variance from background host activity (initial estimate 10–15 hours of active execution time total across scheduled windows, confirmed after measuring decode speed in Phase 1).

---

## 3. Model & Engine Configuration

Because the A9 Mega’s primary competitive advantage is running large models on 128GB unified memory without offloading:

1. **Primary Model:** `Llama-3.3-70B-Instruct` (Q4_K_M quant, ~42GB weights).
   * *Alternative:* `Qwen-2.5-72B-Instruct` (Q4_K_M).
2. **Comparison Model:** `Llama-3.1-8B-Instruct` (Q4_K_M, ~5GB weights) to anchor cross-hardware comparisons against existing Apple M1 (16GB) and RTX 4090 (24GB) results.
3. **Inference Backend:** `llama.cpp-server` (built with ROCm / hipBLAS support for Ryzen AI Max+ 395).
4. **Execution Discipline:** 
   * Single-slot execution (`--parallel 1`, `--no-cont-batching`) to isolate Layer 1 request admission dynamics.
   * Multi-slot test (optional exploratory leg): `--parallel 4` with Clairvoyant's concurrent admission gate (`MAX_CONCURRENCY=4`) to evaluate saturation behavior.

---

## 4. Experimental Arms (Control Matrix)

To scientifically decouple scheduler benefit from proxy overhead, every test runs across three distinct arms:

| Arm | Configuration | Purpose |
| :--- | :--- | :--- |
| **Arm 1: Direct FCFS Baseline** | Client $\to$ `llama.cpp-server` (`:8080`) | Ground truth arrival-order latency (status quo). |
| **Arm 2: Proxy FIFO (Ablation)** | Client $\to$ Clairvoyant Proxy (SJF disabled) $\to$ `llama.cpp-server` | Measures total proxy transit overhead (HTTP hop, JSON parsing; predictor inference itself is 0.029 ms). |
| **Arm 3: Clairvoyant Predictive SJF** | Client $\to$ Clairvoyant Proxy (SJF enabled) $\to$ `llama.cpp-server` | Evaluates predictive reordering, latency reduction, and starvation bounds. |

---

## 5. Workload Profiles & Benchmark Suite

Autoregressive decode on large unified memory models is memory-bandwidth bound, meaning long generations hold the execution slot for substantial durations. Actual decode speed will be directly measured during the Phase 1 smoke test, and run sizes will be calibrated accordingly. Workloads are sized to clearly demonstrate the effect and its direction while respecting physical machine time constraints:

### Workload A: Burst Contention Benchmark
* **70B Model Grid (Scaled for Service Time):**
  * 10 Short prompts ($<200$ tokens) + 10 Long prompts ($>800$ tokens) queued concurrently (enough to show the effect and its direction without excessive machine reservation).
  * 3 repeats per arm (9 runs total; duration determined by measured decode speed, with batch size adjusted further if measured speed is near 4–5 tok/s).
* **8B Model Grid (Full Scale):**
  * 50 Short prompts + 50 Long prompts queued concurrently.
  * 5 repeats per arm (15 runs total).
* **Metrics:** 
  * Short requests: P50, P95, P99 completion latency.
  * Long requests: P50, P95, P99 completion latency.
  * System: Total batch wall-clock time and ordering accuracy (% of Short-Long pairs correctly ordered).

### Workload B: Realistic Poisson Trace Replay (LMSYS Chat Logs)
* **Composition:** 100–150 real-world conversation turns sampled from the LMSYS-Chat-1M corpus.
* **Arrival Process:** Poisson arrivals at queue utilization $\rho \in \{0.60, 0.75\}$, calibrated to empirical mean service time $\mathbb{E}[S]$ on the A9 Mega.
* **Metrics:**
  * Time-To-First-Token (TTFT) P50 and P95.
  * Queue Wait Time P50 and P95.
  * End-to-End Latency P50 and P95.
  * System Throughput (req/s) and completion success rate.

---

## 6. Starvation Timeout ($\tau$) Calibration

In accordance with queueing theory and empirical findings from Section 5.6 of the preprint:
$$\tau = 3 \times \mathbb{E}[S_{\text{short}}]$$
* On the 70B model, $\mathbb{E}[S_{\text{short}}]$ will be empirically measured during the Phase 1 smoke test to calibrate $\tau$ directly on-device.
* Any request waiting in queue longer than $\tau$ is automatically promoted to negative priority values ($< 0.0$), guaranteeing a deterministic upper bound on tail latency for long jobs under high queue saturation.

---

## 7. Reporting & Publication Terms

1. **Independent Evaluation:** The benchmark will be conducted independently. All raw data, latency distributions, and execution logs will be published openly.
2. **Editorial Independence:** GEEKOM may review hardware specifications and system configurations for factual accuracy, but retains zero editorial approval or review over the experimental results, analysis, or publication.
3. **Comprehensive Disclosure:** Findings will report improvements for interactive queries as well as effects on long jobs, tail latency inflation, predictor ranking accuracy, and any edge-case limitations or negative results.
4. **Attribution:** GEEKOM will be credited for providing hardware access on the testbed in any academic publication, preprint update, or technical report.
