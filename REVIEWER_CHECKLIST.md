# Reviewer Checklist — Clairvoyant

Simulated NeurIPS/MLSys review. Use this as a pre-submission gate before v2 (post Vast.ai) and before MLSys 2027 submission.

---

## 🔴 Blocking — Must Fix Before Any Submission

### 1. End-to-End Latency Results (§6.3 — "in preparation")
- [ ] Run Vast.ai benchmark: RTX 4090, Gemma3:4b + Llama3.1:8b, 5 runs per condition
- [ ] Report FCFS vs. SJF: P50 / P95 / P99 for short-class requests
- [ ] Minimum: 100 mixed-workload requests, single GPU
- [ ] If blocked: reframe as "predictor design study" and move latency claims to future work

### 2. Ranking Metric — Medium-Class Boundary Sensitivity
- [ ] Add sensitivity analysis: does ranking accuracy hold if boundaries shift to 150/850 or 250/750 tokens?
- [ ] Report weighted metric including Medium examples (partial credit for near-correct ordering)
- [ ] Clarify: how are Medium requests handled in end-to-end scheduling?

### 3. Simple Baselines Missing
- [ ] Add baseline table: FCFS / prompt-length threshold / keyword heuristic / Clairvoyant
- [ ] This either strengthens the "lexical sufficiency" claim or clarifies the ML contribution

### 4. Cross-Distribution Performance (~60%) — Production Justification
- [ ] Add queueing-theory analysis: at 60% pairwise accuracy + realistic workload mix, what is expected reduction in mean wait time vs. FCFS?
- [ ] Add guidance: "If workload distribution shifts by X%, retrain the predictor"

---

## 🟡 Moderate — Strengthen for Camera-Ready

### 5. Token Approximation (`len // 4`) Error Analysis
- [ ] Compare `//4` vs. actual tokeniser counts on a subset of each dataset
- [ ] Add to limitations: "Deployments should use backend tokeniser for label generation where possible"

### 6. Starvation Timeout τ Sensitivity
- [ ] Add plot: scheduling quality vs. τ multiplier (1×, 2×, 3×, 5×)
- [ ] Discuss auto-calibration strategies (online estimation of μ_short)

### 7. Adversarial / Gaming Considerations
- [ ] Brief discussion: can users game the scheduler in multi-tenant deployments?
- [ ] Consider: confidence threshold — if P(Long) ≈ 0.5, fall back to FCFS

### 8. English-Only Limitation
- [ ] Move language limitation higher in §7 (Limitations)
- [ ] Suggest path forward: language detection + language-specific verb sets, or embedding-based fallback

---

## ✅ Confirmed Strengths (Keep and Emphasise)

- Dataset finding: Alpaca/CodeAlpaca unusable for SJF due to GPT-imposed brevity — novel, citable
- Ranking vs. classification gap (+21–29pp): validates continuous P(Long) as scheduling key
- Scope delineation: serial/low-concurrency vs. continuous batching — helps practitioners
- Reproducibility: open-source, public datasets, standard tooling
- Architecture simplicity: 0.029ms sidecar overhead

---

## Version Gate

| Item | Required for arXiv v1 | Required for MLSys 2027 |
|---|---|---|
| §6.3 GPU benchmarks | ❌ (future work) | ✅ |
| Baseline table | Recommended | ✅ |
| Medium-class sensitivity | Recommended | ✅ |
| τ sensitivity plot | Optional | ✅ |
| Queueing-theory analysis | Optional | ✅ |
| Adversarial discussion | Optional | Recommended |
