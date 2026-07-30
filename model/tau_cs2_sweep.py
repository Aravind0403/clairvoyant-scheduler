"""
τ sensitivity across C_s² profiles — addresses reviewer R5.

Sweeps starvation timeout τ at three service-time variance profiles:
  C_s² = 0.5  (low variance)
  C_s² = 1.0  (exponential — memoryless)
  C_s² = 2.0  (high variance, heavy-tailed; close to measured LLM distribution)

Service times drawn from Gamma(k, θ) where k=1/C_s², θ=µ×C_s²,
so E[X]=µ and Var[X]=µ²×C_s² by construction.

Fixed parameters (calibrated to RTX 4090 Gemma3:4b, consistent with main DES):
  µ_short = 3.5s,  µ_long = 8.9s,  50/50 mix,  ρ = 0.74,  n = 2000,  5 seeds

Output:
  results/tau_cs2_sweep.json   — raw numbers for LaTeX table
  results/tau_cs2_sweep.png    — Pareto frontier per C_s² profile

Usage:
    python model/tau_cs2_sweep.py
"""

import heapq
import json
import pathlib
import statistics

import numpy as np

# ── Fixed parameters ───────────────────────────────────────────────────────────
MU_SHORT   = 3.5
MU_LONG    = 8.9
SHORT_FRAC = 0.5
E_S        = SHORT_FRAC * MU_SHORT + (1 - SHORT_FRAC) * MU_LONG   # 6.2s
LAMBDA     = 0.74 / E_S                                             # ≈ 0.1194/s
N_REQ      = 2000
N_SEEDS    = 5

CS2_PROFILES = [0.5, 1.0, 2.0]
TAU_MULTS    = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")]


# ── Gamma service-time sampler ────────────────────────────────────────────────

def gamma_service(rng, mu, cs2, n):
    """Sample n Gamma(k, theta) service times with mean=mu, C_s²=cs2."""
    k     = 1.0 / cs2
    theta = mu * cs2
    return rng.gamma(k, theta, n)


# ── Core DES: SJF with starvation timeout ─────────────────────────────────────

def simulate_sjf(tau_sec, cs2, seed):
    rng = np.random.default_rng(seed)

    inter = rng.exponential(1.0 / LAMBDA, N_REQ)
    arrivals = np.cumsum(inter)

    is_short = rng.random(N_REQ) < SHORT_FRAC
    svc = np.where(
        is_short,
        gamma_service(rng, MU_SHORT, cs2, N_REQ),
        gamma_service(rng, MU_LONG,  cs2, N_REQ),
    )
    # Predictor: noisy P(Long) signal (matches existing DES)
    true_p = np.where(is_short, 0.1, 0.9)
    p_long = np.clip(true_p + rng.normal(0, 0.15, N_REQ), 0.0, 1.0)

    queue   = []   # (p_long, arrival_time, idx, svc)
    done_short, done_long = [], []
    t = 0.0
    nxt = 0

    def flush_arrivals(upto):
        nonlocal nxt
        while nxt < N_REQ and arrivals[nxt] <= upto:
            heapq.heappush(queue, (p_long[nxt], arrivals[nxt], nxt, svc[nxt]))
            nxt += 1

    while len(done_short) + len(done_long) < N_REQ:
        flush_arrivals(t)

        if not queue:
            if nxt < N_REQ:
                t = arrivals[nxt]
                continue
            break

        # Starvation check: promote oldest waiter if waiting > tau
        if tau_sec != float("inf"):
            for i, (pl, at, idx, s) in enumerate(queue):
                if t - at > tau_sec:
                    queue[i] = (-1.0, at, idx, s)
                    heapq.heapify(queue)
                    break

        _, at, idx, s = heapq.heappop(queue)
        start   = max(t, arrivals[idx])
        finish  = start + s
        sojourn = finish - arrivals[idx]
        t = finish
        (done_short if is_short[idx] else done_long).append(sojourn)
        flush_arrivals(t)

    def p(data, pct):
        return float(np.percentile(data, pct)) if data else float("nan")

    return {
        "short_p50": p(done_short, 50),
        "short_p95": p(done_short, 95),
        "long_p50":  p(done_long,  50),
        "long_p95":  p(done_long,  95),
    }


def simulate_fcfs(cs2, seed):
    rng = np.random.default_rng(seed + 999)

    inter = rng.exponential(1.0 / LAMBDA, N_REQ)
    arrivals = np.cumsum(inter)
    is_short = rng.random(N_REQ) < SHORT_FRAC
    svc = np.where(
        is_short,
        gamma_service(rng, MU_SHORT, cs2, N_REQ),
        gamma_service(rng, MU_LONG,  cs2, N_REQ),
    )

    queue = []
    done_short, done_long = [], []
    t = 0.0
    nxt = 0

    while len(done_short) + len(done_long) < N_REQ:
        while nxt < N_REQ and arrivals[nxt] <= t:
            heapq.heappush(queue, (arrivals[nxt], nxt))
            nxt += 1
        if not queue:
            if nxt < N_REQ:
                t = arrivals[nxt]
                continue
            break
        _, idx = heapq.heappop(queue)
        start   = max(t, arrivals[idx])
        finish  = start + svc[idx]
        sojourn = finish - arrivals[idx]
        t = finish
        (done_short if is_short[idx] else done_long).append(sojourn)
        while nxt < N_REQ and arrivals[nxt] <= t:
            heapq.heappush(queue, (arrivals[nxt], nxt))
            nxt += 1

    return {
        "short_p50": float(np.percentile(done_short, 50)),
        "short_p95": float(np.percentile(done_short, 95)),
        "long_p50":  float(np.percentile(done_long,  50)),
        "long_p95":  float(np.percentile(done_long,  95)),
    }


# ── Sweep and report ───────────────────────────────────────────────────────────

def mean_over_seeds(fn, **kwargs):
    rows = [fn(**kwargs, seed=s) for s in range(N_SEEDS)]
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def run_sweep():
    all_results = {}

    for cs2 in CS2_PROFILES:
        print(f"\n{'='*60}")
        print(f"C_s² = {cs2}  (σ_short={MU_SHORT*cs2**0.5:.2f}s, "
              f"σ_long={MU_LONG*cs2**0.5:.2f}s)")
        print(f"{'='*60}")

        fcfs = mean_over_seeds(simulate_fcfs, cs2=cs2)
        print(f"FCFS  SHORT p50={fcfs['short_p50']:6.2f}s  "
              f"p95={fcfs['short_p95']:6.2f}s  "
              f"LONG p50={fcfs['long_p50']:6.2f}s")

        tau_rows = []
        for mult in TAU_MULTS:
            tau_sec = mult * MU_SHORT if mult != float("inf") else float("inf")
            r = mean_over_seeds(simulate_sjf, tau_sec=tau_sec, cs2=cs2)
            red = (fcfs["short_p50"] - r["short_p50"]) / fcfs["short_p50"] * 100
            label = f"{mult}×" if mult != float("inf") else "∞"
            print(f"τ={label:<5}  SHORT p50={r['short_p50']:6.2f}s "
                  f"({red:+.1f}%)  "
                  f"p95={r['short_p95']:6.2f}s  "
                  f"LONG p50={r['long_p50']:6.2f}s")
            tau_rows.append({"tau_mult": mult, "tau_sec": tau_sec,
                             "reduction_pct": round(red, 1), **r})

        all_results[str(cs2)] = {"fcfs": fcfs, "sjf": tau_rows}

    return all_results


def save_results(results, out_dir):
    path = out_dir / "tau_cs2_sweep.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {path}")


def plot_results(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"0.5": "#2E86AB", "1.0": "#A23B72", "2.0": "#F18F01"}
        labels = {"0.5": r"$C_s^2=0.5$ (low variance)",
                  "1.0": r"$C_s^2=1.0$ (exponential)",
                  "2.0": r"$C_s^2=2.0$ (high variance)"}

        fig, ax = plt.subplots(figsize=(7, 5))

        for cs2_key, data in results.items():
            fcfs_short = data["fcfs"]["short_p50"]
            fcfs_long  = data["fcfs"]["long_p50"]
            sjf_short  = [r["short_p50"] for r in data["sjf"]]
            sjf_long   = [r["long_p50"]  for r in data["sjf"]]

            ax.plot(sjf_short, sjf_long,
                    "o-", color=colors[cs2_key],
                    label=labels[cs2_key], linewidth=2, markersize=6)

            # Annotate τ=3× point
            tau3_idx = TAU_MULTS.index(3.0)
            ax.annotate(r"$\tau=3\times$",
                        (sjf_short[tau3_idx], sjf_long[tau3_idx]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=colors[cs2_key])

            # FCFS point
            ax.scatter([fcfs_short], [fcfs_long],
                       color=colors[cs2_key], marker="*", s=140, zorder=6)

        ax.set_xlabel("Short-request P50 sojourn time (s)", fontsize=11)
        ax.set_ylabel("Long-request P50 sojourn time (s)", fontsize=11)
        ax.set_title(r"Pareto frontier across $C_s^2$ profiles ($\rho=0.74$)",
                     fontsize=12)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.25)

        fig.text(0.01, 0.01,
                 f"µ_short={MU_SHORT}s, µ_long={MU_LONG}s, ρ=0.74, n={N_REQ}, 5 seeds. "
                 r"Stars = FCFS baseline.",
                 fontsize=7, color="gray")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path = out_dir / "tau_cs2_sweep.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


def main():
    out_dir = pathlib.Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    results = run_sweep()
    save_results(results, out_dir)
    plot_results(results, out_dir)

    # Print LaTeX table snippet for paper
    print("\n\n=== LaTeX table rows (SHORT P50, τ=3×µ_short) ===")
    print(r"\midrule")
    for cs2_key, data in results.items():
        tau3 = next(r for r in data["sjf"] if r["tau_mult"] == 3.0)
        fcfs = data["fcfs"]["short_p50"]
        sjf  = tau3["short_p50"]
        red  = tau3["reduction_pct"]
        print(f"$C_s^2={cs2_key}$ & {fcfs:.2f} & {sjf:.2f} & {red:.1f}\\% \\\\")


if __name__ == "__main__":
    main()
