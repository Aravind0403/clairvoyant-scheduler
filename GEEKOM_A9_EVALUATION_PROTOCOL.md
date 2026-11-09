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
  * Phase 1: 30–45 minute smoke test (environment verification, ROCm check, single-prompt end-to-end baseline).
  * Phase 2: Two 2-hour benchmark windows scheduled during off-peak hours (UTC+8) to eliminate latency variance from background host activity. Total active machine time: $< 6$ hours.

---

## 3. Model & Engine Configuration

Because the A9 Mega’s primary competitive advantage is running large models on 128GB unified memory without offloading:

1. **Primary Model:** `Llama-3.3-70B-Instruct` (Q4_K_M quant, ~42GB weights).
   * *Alternative:* `Qwen-2.5-72B-Instruct` (Q4_K_M).
2. **Comparison Model:** `Llama-3.1-8B-Instruct` (Q4_K_M, ~5GB weights) to anchor cross-hardware comparisons against existing Apple M1 (16GB) and RTX 4090 (24GB) results.
3. **Inference Backend:** `llama.cpp-server` (built with ROCm / hipBLAS support for Ryzen AI Max+ 395).
4. **Execution Discipline:** 
   * Single-slot execution (`--parallel 1`, `--cont-batching false`) to isolate Layer 1 request admission dynamics.
   * Multi-slot test (optional exploratory leg): `--parallel 4` with Clairvoyant's concurrent admission gate (`MAX_CONCURRENCY=4`) to evaluate saturation behavior.

---

## 4. Experimental Arms (Control Matrix)

To scientifically decouple scheduler benefit from proxy overhead, every test runs across three distinct arms:

| Arm | Configuration | Purpose |
| :--- | :--- | :--- |
| **Arm 1: Direct FCFS Baseline** | Client $\to$ `llama.cpp-server` (`:8080`) | Ground truth arrival-order latency (status quo). |
| **Arm 2: Proxy FIFO (Ablation)** | Client $\to$ Clairvoyant Proxy (SJF disabled) $\to$ `llama.cpp-server` | Measures proxy overhead ($<0.029\text{ ms}$). |
| **Arm 3: Clairvoyant Predictive SJF** | Client $\to$ Clairvoyant Proxy (SJF enabled) $\to$ `llama.cpp-server` | Evaluates predictive reordering and starvation mitigation. |

---

## 5. Workload Profiles & Benchmark Suite

### Workload A: Burst Contention Benchmark (100 Requests)
* **Composition:** 50 Short prompts ($<200$ expected tokens) + 50 Long prompts ($>800$ expected tokens) queued simultaneously.
* **Metrics:** 
  * Short requests: P50, P95, P99 completion latency.
  * Long requests: P50, P95, P99 completion latency.
  * System: Total batch wall-clock time and ordering accuracy (% of Short-Long pairs correctly ordered).
* **Repeats:** 5 independent runs per arm (mean $\pm$ 1 std dev).

### Workload B: Realistic Poisson Trace Replay (LMSYS Chat Logs)
* **Composition:** 150 real-world conversation turns sampled from the LMSYS-Chat-1M corpus.
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
* On the 70B model, $\mathbb{E}[S_{\text{short}}]$ will be empirically measured during the Phase 1 smoke test.
* Any request waiting in queue longer than $\tau$ is automatically promoted to priority 0, providing a deterministic upper bound on tail latency for long jobs under high queue saturation.

---

## 7. Reporting & Publication Terms

1. **Independent Evaluation:** The benchmark will be conducted independently. All raw data, latency distributions, and logs will be published openly.
2. **Comprehensive Disclosure:** Findings will report improvements for interactive queries as well as effects on long jobs, tail latency inflation, predictor ranking accuracy, and any edge-case limitations or negative results.
3. **Attribution:** GEEKOM will be credited for providing hardware access on the testbed in any academic publication, preprint update, or technical report.

