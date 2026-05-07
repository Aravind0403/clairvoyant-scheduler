# Clairvoyant Scheduler — Research & Publication Roadmap

**Target Venue:** MLSys 2027 (Submission deadline ~October 2026)
**Immediate Goal:** Post to arXiv (June 2026) for job search priority.

This roadmap outlines the step-by-step plan to take the current Clairvoyant Scheduler architecture (v0.55) from a working prototype to a published systems research paper.

---

## 1. May 2026: Draft the Paper

**Objective:** Write the core paper using the existing Phase 4 data. Do not wait for new experiments to start writing.
**Output:** 10–12 page draft.

**Structure:**
- **Abstract**
- **Introduction:** The Head-of-Line Blocking (HOLB) problem in LLM serving.
- **Related Work:** FCFS limitations, Orca, continuous batching (vLLM), S3 (longest-first throughput).
- **System Design:** The sidecar architecture, predictive ML model (<0.029ms overhead), and lexical feature extraction.
- **Evaluation:** Present the Ollama (M1 & RTX 4090) results proving 100% SJF ordering and 44% latency reduction for short requests. *Crucially*, include the vLLM Stage 3 negative result (serial dispatcher harms batching) to establish operational boundaries.
- **Discussion/Conclusion**

---

## 2. June 2026: Ablation Study & arXiv Submission

**Objective:** Establish what features drive the ML predictor and secure a public timestamp for job applications.

**Action Items:**
- [ ] **Ablation Study:** Write a script using `model/train.py` to drop one feature at a time from `data/training_data.csv`. Retrain the XGBoost model and measure the drop in the 76.29% ranking accuracy.
- [ ] Identify which of the 19 lexical features (e.g., `has_code_keyword`, token length) carry the most weight.
- [ ] **Submit to arXiv:** Once the draft includes the ablation study and current benchmarks, upload to arXiv.
- [ ] **Job Search:** Add the arXiv link to CV and LinkedIn immediately.

---

## 3. July 2026: Oracle SJF Baseline

**Objective:** Establish the theoretical maximum advantage of the system.

**Action Items:**
- [ ] Run benchmarks using ground-truth output length labels instead of the ML-predicted labels.
- [ ] Calculate the ordering accuracy and latency reduction under perfect prediction conditions.
- [ ] Compare current ML performance against this Oracle bound to show how much headroom remains.

---

## 4. August 2026: Proper vLLM Benchmark (The Concurrent Dispatcher)

**Objective:** Turn the Stage 3 vLLM "ceiling comparison" into a major system contribution by adapting the dispatcher for continuous batching engines.

**Action Items:**
- [ ] **Architectural Tweak:** Modify the Go scheduler (`scheduler/proxy/dispatcher.go`). Instead of popping one request serially, implement a priority admission gate that pops a batch of the $N$ shortest requests concurrently.
- [ ] **Vast.ai Benchmark:** Spin up a new RTX 4090 instance.
- [ ] **Workload:** Use a 13B model under sustained load (Poisson arrivals) hitting 60–70% GPU utilization. Mix short prompts with long RAG-style prompts (1000–2000 tokens).
- [ ] **Metrics:** Measure P99 Time-To-First-Token (TTFT) comparing standard vLLM FCFS vs. vLLM with Clairvoyant Admission Control.

---

## 5. September 2026: M/G/1 Theoretical Frame

**Objective:** Elevate the paper from pure engineering to robust systems research by grounding it in queueing theory.

**Action Items:**
- [ ] Model the system as a non-preemptive priority queue.
- [ ] Derive the expected P99 latency gain as a function of:
  - Arrival rate ($\lambda$)
  - Short request service rate ($\mu_{short}$)
  - Long request service rate ($\mu_{long}$)
  - Mixing ratio ($R$)
- [ ] Add the 1–2 pages of mathematical analysis to the paper draft. *(Note: Consider collaborating with a queueing theory specialist if needed).*

---

## 6. October 2026: MLSys 2027 Submission

**Objective:** Finalize and submit.

**Action Items:**
- [ ] Polish and proofread the manuscript.
- [ ] Format to the official MLSys double-blind template.
- [ ] Submit to MLSys.
- [ ] Update the arXiv preprint with the final, complete version (including the new vLLM benchmarks and M/G/1 theory).

---

## Post-Submission: January 2027

- **Author Response Period:** Address reviewer concerns (the ablation study, baselines, and theoretical framing completed in previous months will form a strong defense).
