# Clairvoyant — Paper Review Notes

**Title (final):** Clairvoyant: Predictive SJF Scheduling to Mitigate Head-of-Line Blocking in Serial LLM Backends

---

## Strengths

**1. Scope discipline is excellent**
Clearly constrained to serial inference backends, low concurrency, edge/local deployments, Layer-1 HOLB. Avoids reviewer backlash from overclaiming.

**2. Queueing-theory framing is solid**
M/G/1 + HOLB explanation materially upgrades the paper. Especially strong: why FCFS fails under high variance, why approximate ranking is sufficient, why pairwise ordering matters more than exact prediction. Makes the system feel principled rather than heuristic.

**3. Ranking over classification argument is genuinely good**
One of the strongest intellectual parts. Evaluation metric aligned with operational goal — that is systems-thinking.

**4. Dataset-starvation finding is publishable-quality insight**
Probably the most novel conceptual contribution. The observation that instruction datasets are structurally unsuitable for scheduling prediction is concrete, measurable, reproducible, and operationally meaningful.

**5. Predictor design is appropriately lightweight**
ONNX + lexical feature framing works: low overhead is measurable, edge deployment constraint is believable, embedding latency comparison strengthens the argument.

**6. Paper reads like systems work**
Explains deployment regimes, tradeoffs, boundaries, and where Clairvoyant loses. That maturity helps a lot.

---

## Gaps to Address (in priority order)

**1. Evaluation depth — biggest weakness**
Single-GPU emphasis, limited workload realism, relatively small-scale benchmarks, no long-running experiments, limited concurrency exploration. Currently proves "the idea works" but not "this system is comprehensively validated." Target: v2.

**2. No deployment-boundary quantification**
Missing: KV-cache memory curves, concurrency vs VRAM plots, consumer GPU feasibility analysis, "where Clairvoyant wins vs batching" chart. High-impact missing section.

**3. Visual presentation is underdeveloped**
No polished latency CDFs, no queue progression charts, no visually striking result. Most successful systems papers have one graph reviewers remember. Candidates: latency CDF separation, utilisation vs tail latency, queue collapse visualisation.

**4. Empirical scale feels workshop-tier**
Missing: trace replay, mixed real workloads, queue occupancy analysis, latency-throughput tradeoff curves, sustained runtime experiments, TTFT analysis, scheduler overhead under load.

**5. Repetition — remove 10–15%**
Serial inference scope, Layer-1 vs Layer-2 distinction, and batching comparison are repeated multiple times. Can be trimmed without losing clarity.

**6. Limited baseline comparisons**
Currently compares against FCFS, heuristics, and embeddings. Reviewers may ask: why not static size estimation? Token-count heuristics? FIFO+aging? Even weak baselines help calibrate improvement.

**7. Scheduler novelty is moderate — evaluation must compensate**
Novelty is in deployment framing, lightweight predictor, serial inference applicability — not in inventing SJF. This means evaluation quality, systems rigour, and deployment realism must carry more weight.

**8. No operational observability section**
Missing: metrics exported, queue telemetry, debugging behaviour, scheduling introspection, failure handling, backpressure behaviour. A short "Operational Considerations" section would help with infra audiences.

**9. Starvation analysis needs deeper treatment**
Current timeout discussion is good but could be stronger with percentile starvation bounds, fairness curves, sensitivity sweep visualisations, throughput/fairness tradeoff graphs.

**10. Need one memorable systems graph**
Candidates: HOLB explosion under FCFS vs Clairvoyant, latency CDF separation, utilisation vs tail latency, deployment-boundary VRAM graph, queue collapse visualisation.

---

---

## External Reviewer Assessment (Jun 2026)

**Venue verdict:**

| Venue | Assessment |
|-------|-----------|
| NeurIPS / MLSys Main | Reject |
| NeurIPS / MLSys Workshop | Borderline → Likely Accept with one experiment |
| Industry / Engineering Workshop | Strong Accept |

**Scores:** Novelty 5/10 · Soundness 7/10 · Experiments 6/10 · Impact 8/10 · Writing 8.5/10 · Overall 6.5–7/10

**Top weaknesses:**
1. Limited novelty — SJF + lightweight predictor is a known combination; deployment framing is the differentiator
2. Single hardware platform (RTX 4090 only)
3. Baselines weak — missing LTR, embedding-based, proxy-model comparisons
4. No TTFT / queue-wait decomposition — only end-to-end latency reported
5. Workload is synthetic burst — no trace replay

**Highest-impact single experiment:**
Realistic workload replay (500–2000 prompts from LMSYS/ShareGPT, natural length distribution) under ρ ∈ {0.4, 0.6, 0.8}, reporting TTFT P50/P95, Queue Wait P50/P95, End-to-End P50/P95 for FCFS vs Clairvoyant. Addresses three reviewer concerns at once.

**Second experiment:** One additional backend (llama.cpp CPU-only, RTX 3060, or Apple Silicon).

**Do NOT spend time on:** feature ablations, SHAP plots, XGBoost tuning, more datasets, larger models.

**Expected outcome after both experiments:** NeurIPS/MLSys Workshop 70–85% acceptance probability.

*Last updated: 2026-06-04*
