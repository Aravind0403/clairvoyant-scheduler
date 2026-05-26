"""
τ sensitivity analysis + workload-spectrum sweep for Clairvoyant.

Two outputs
-----------
1. τ sensitivity table  — fixed ρ=0.74, vary τ (as before)
2. Workload spectrum    — fixed τ=3×μ_short, vary ρ ∈ {0.30, 0.50, 0.60, 0.74, 0.85}
   Generates results/workload_spectrum.json and results/workload_spectrum.png

Service time parameters derived from Gemma3:4b RTX 4090 benchmark:
  Short: mean=3.5s, std=0.8s
  Long:  mean=8.9s, std=2.0s
  Mix:   50% short, 50% long
  E[S]   = 0.5×3.5 + 0.5×8.9 = 6.2s

Usage:
    python model/tau_sensitivity.py
"""

import heapq
import json
import random
import statistics
import numpy as np
import pathlib

random.seed(42)
np.random.seed(42)

# ── Service time parameters (RTX 4090 Gemma3:4b) ──────────────────────────────
SHORT_MEAN = 3.5
SHORT_STD  = 0.8
LONG_MEAN  = 8.9
LONG_STD   = 2.0
SHORT_FRAC = 0.5
E_S        = SHORT_FRAC * SHORT_MEAN + (1 - SHORT_FRAC) * LONG_MEAN  # 6.2s

# ── Default simulation parameters ─────────────────────────────────────────────
ARRIVAL_RATE = 0.12   # λ at ρ≈0.74
N_REQUESTS   = 2000
N_SEEDS      = 5

MU_SHORT         = SHORT_MEAN
TAU_MULTIPLIERS  = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")]

# ── ρ values for workload-spectrum sweep ──────────────────────────────────────
RHO_VALUES = [0.30, 0.50, 0.60, 0.74, 0.85]


# ── Request class ─────────────────────────────────────────────────────────────

class Request:
    def __init__(self, req_id, is_short, arrival_time, service_time, p_long):
        self.req_id       = req_id
        self.is_short     = is_short
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.p_long       = p_long
        self.start_time   = None
        self.finish_time  = None

    @property
    def wait_time(self):
        return self.start_time - self.arrival_time

    @property
    def sojourn_time(self):
        return self.finish_time - self.arrival_time


# ── Core simulators (arrival_rate parameterised) ──────────────────────────────

def simulate(tau_sec: float, seed: int,
             arrival_rate: float = ARRIVAL_RATE,
             n_requests: int = N_REQUESTS) -> dict:
    """SJF with starvation timeout τ, Poisson arrivals."""
    rng = np.random.default_rng(seed)

    inter_arrivals = rng.exponential(1.0 / arrival_rate, n_requests)
    arrivals = np.cumsum(inter_arrivals)

    is_short_arr  = rng.random(n_requests) < SHORT_FRAC
    service_times = np.where(
        is_short_arr,
        np.abs(rng.normal(SHORT_MEAN, SHORT_STD, n_requests)),
        np.abs(rng.normal(LONG_MEAN,  LONG_STD,  n_requests)),
    )
    true_p = np.where(is_short_arr, 0.1, 0.9)
    noise  = rng.normal(0, 0.15, n_requests)
    p_long = np.clip(true_p + noise, 0.0, 1.0)

    requests = [
        Request(i, is_short_arr[i], arrivals[i], service_times[i], p_long[i])
        for i in range(n_requests)
    ]

    waiting_queue = []
    completed     = []
    dispatch_time = 0.0
    next_idx      = 0

    def enqueue(req):
        heapq.heappush(waiting_queue, (req.p_long, req.arrival_time, req.req_id, req))

    def check_starvation(t):
        if tau_sec == float("inf") or not waiting_queue:
            return
        for i, (pl, at, rid, req) in enumerate(waiting_queue):
            if t - at > tau_sec:
                waiting_queue[i] = (-1.0, at, rid, req)
                heapq.heapify(waiting_queue)
                break

    while len(completed) < n_requests:
        while next_idx < n_requests and requests[next_idx].arrival_time <= dispatch_time:
            enqueue(requests[next_idx])
            next_idx += 1

        if not waiting_queue:
            if next_idx < n_requests:
                dispatch_time = requests[next_idx].arrival_time
                continue
            else:
                break

        check_starvation(dispatch_time)
        _, _, _, req = heapq.heappop(waiting_queue)
        req.start_time  = max(dispatch_time, req.arrival_time)
        req.finish_time = req.start_time + req.service_time
        dispatch_time   = req.finish_time
        completed.append(req)

        while next_idx < n_requests and requests[next_idx].arrival_time <= dispatch_time:
            enqueue(requests[next_idx])
            next_idx += 1

    short_sojourn = [r.sojourn_time for r in completed if r.is_short]
    long_sojourn  = [r.sojourn_time for r in completed if not r.is_short]

    def pct(data, p):
        return float(np.percentile(data, p)) if data else float("nan")

    return {
        "short_p50": pct(short_sojourn, 50),
        "short_p95": pct(short_sojourn, 95),
        "short_p99": pct(short_sojourn, 99),
        "long_p50":  pct(long_sojourn,  50),
        "long_p95":  pct(long_sojourn,  95),
        "long_p99":  pct(long_sojourn,  99),
    }


def simulate_fcfs(seed: int,
                  arrival_rate: float = ARRIVAL_RATE,
                  n_requests: int = N_REQUESTS) -> dict:
    """FCFS baseline (arrival-order priority queue)."""
    rng = np.random.default_rng(seed + 100)

    inter_arrivals = rng.exponential(1.0 / arrival_rate, n_requests)
    arrivals = np.cumsum(inter_arrivals)
    is_short_arr  = rng.random(n_requests) < SHORT_FRAC
    service_times = np.where(
        is_short_arr,
        np.abs(rng.normal(SHORT_MEAN, SHORT_STD, n_requests)),
        np.abs(rng.normal(LONG_MEAN,  LONG_STD,  n_requests)),
    )
    requests = [
        Request(i, is_short_arr[i], arrivals[i], service_times[i], float(i))
        for i in range(n_requests)
    ]

    waiting_queue = []
    completed     = []
    dispatch_time = 0.0
    next_idx      = 0

    while len(completed) < n_requests:
        while next_idx < n_requests and requests[next_idx].arrival_time <= dispatch_time:
            r = requests[next_idx]
            heapq.heappush(waiting_queue, (r.arrival_time, r.req_id, r))
            next_idx += 1
        if not waiting_queue:
            if next_idx < n_requests:
                dispatch_time = requests[next_idx].arrival_time
                continue
            else:
                break
        _, _, req = heapq.heappop(waiting_queue)
        req.start_time  = max(dispatch_time, req.arrival_time)
        req.finish_time = req.start_time + req.service_time
        dispatch_time   = req.finish_time
        completed.append(req)
        while next_idx < n_requests and requests[next_idx].arrival_time <= dispatch_time:
            r = requests[next_idx]
            heapq.heappush(waiting_queue, (r.arrival_time, r.req_id, r))
            next_idx += 1

    short_sojourn = [r.sojourn_time for r in completed if r.is_short]
    long_sojourn  = [r.sojourn_time for r in completed if not r.is_short]
    return {
        "short_p50": float(np.percentile(short_sojourn, 50)),
        "short_p95": float(np.percentile(short_sojourn, 95)),
        "long_p50":  float(np.percentile(long_sojourn,  50)),
        "long_p95":  float(np.percentile(long_sojourn,  95)),
    }


# ── τ sensitivity sweep (original analysis) ───────────────────────────────────

def tau_sensitivity_main():
    results = []

    print(f"=== τ Sensitivity (ρ≈0.74, λ={ARRIVAL_RATE}/s) ===")
    print(f"Short: N({SHORT_MEAN}s, {SHORT_STD}s)  "
          f"Long: N({LONG_MEAN}s, {LONG_STD}s)  Mix: {SHORT_FRAC:.0%}")
    print()

    for tau_mult in TAU_MULTIPLIERS:
        tau_sec = tau_mult * MU_SHORT if tau_mult != float("inf") else float("inf")
        seed_results = [simulate(tau_sec, s) for s in range(N_SEEDS)]

        row = {
            "tau_mult":  tau_mult,
            "tau_sec":   tau_sec,
            "short_p50": statistics.mean(r["short_p50"] for r in seed_results),
            "short_p95": statistics.mean(r["short_p95"] for r in seed_results),
            "long_p50":  statistics.mean(r["long_p50"]  for r in seed_results),
            "long_p95":  statistics.mean(r["long_p95"]  for r in seed_results),
        }
        results.append(row)

        tau_label = f"{tau_mult}×" if tau_mult != float("inf") else "∞"
        print(f"τ={tau_label:<6}  SHORT p50={row['short_p50']:6.2f}s  "
              f"p95={row['short_p95']:6.2f}s  "
              f"LONG p50={row['long_p50']:6.2f}s  p95={row['long_p95']:6.2f}s")

    fcfs_runs = [simulate_fcfs(s) for s in range(N_SEEDS)]
    fcfs = {k: statistics.mean(r[k] for r in fcfs_runs) for k in fcfs_runs[0]}
    print(f"\nFCFS baseline:  SHORT p50={fcfs['short_p50']:6.2f}s  "
          f"p95={fcfs['short_p95']:6.2f}s  "
          f"LONG p50={fcfs['long_p50']:6.2f}s  p95={fcfs['long_p95']:6.2f}s")

    return results, fcfs


# ── ρ sweep for workload spectrum ─────────────────────────────────────────────

def rho_sweep_main():
    """Sweep ρ ∈ RHO_VALUES at τ=3×μ_short. Returns list of dicts."""
    tau_sec = 3.0 * MU_SHORT   # 10.5s — recommended default

    print(f"\n=== Workload Spectrum Sweep (τ=3×μ_short={tau_sec}s) ===")
    print(f"{'ρ':>6}  {'λ (1/s)':>9}  "
          f"{'FCFS P50':>10}  {'SJF P50':>9}  {'Reduction':>10}  {'±std':>6}")
    print("-" * 62)

    spectrum = []

    for rho in RHO_VALUES:
        lam = rho / E_S   # λ = ρ / E[S]

        # SJF seeds
        sjf_p50_per_seed  = [simulate(tau_sec, s, arrival_rate=lam)["short_p50"]
                              for s in range(N_SEEDS)]
        # FCFS seeds
        fcfs_p50_per_seed = [simulate_fcfs(s, arrival_rate=lam)["short_p50"]
                              for s in range(N_SEEDS)]

        sjf_p50_mean  = statistics.mean(sjf_p50_per_seed)
        fcfs_p50_mean = statistics.mean(fcfs_p50_per_seed)
        reduction     = (fcfs_p50_mean - sjf_p50_mean) / fcfs_p50_mean * 100

        # Propagated std: std of per-seed reduction values
        per_seed_reduction = [
            (f - s) / f * 100
            for f, s in zip(fcfs_p50_per_seed, sjf_p50_per_seed)
        ]
        reduction_std = statistics.stdev(per_seed_reduction) if len(per_seed_reduction) > 1 else 0.0

        row = {
            "rho":           rho,
            "lambda":        round(lam, 4),
            "fcfs_p50":      round(fcfs_p50_mean, 2),
            "sjf_p50":       round(sjf_p50_mean, 2),
            "reduction_pct": round(reduction, 1),
            "reduction_std": round(reduction_std, 1),
        }
        spectrum.append(row)

        print(f"{rho:>6.2f}  {lam:>9.4f}  "
              f"{fcfs_p50_mean:>10.2f}s  {sjf_p50_mean:>9.2f}s  "
              f"{reduction:>9.1f}%  ±{reduction_std:.1f}%")

    return spectrum


# ── Figure generation ─────────────────────────────────────────────────────────

def plot_workload_spectrum(spectrum: list, out_dir: pathlib.Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rhos       = [r["rho"]           for r in spectrum]
        reductions = [r["reduction_pct"] for r in spectrum]
        errs       = [r["reduction_std"] for r in spectrum]

        # Burst point from RTX 4090 GPU benchmark (Gemma3:4b, n=250)
        burst_rho       = 1.10   # plotted right of ρ=1 to visually separate
        burst_reduction = 70.0   # SHORT P50: 229.5s → 69.1s

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Simulation curve
        ax.errorbar(rhos, reductions, yerr=errs,
                    fmt="o-", color="#2E86AB", linewidth=2, markersize=7,
                    capsize=4, label="Poisson arrivals (DES, 5 seeds, ±1 std dev)")

        # Burst point
        ax.scatter([burst_rho], [burst_reduction],
                   color="#A23B72", s=110, zorder=6,
                   label="Concurrent burst, RTX 4090 (n=250, no CI)")
        ax.annotate("Burst\n(upper bound)",
                    (burst_rho, burst_reduction),
                    textcoords="offset points", xytext=(6, -18), fontsize=8,
                    color="#A23B72")

        # Annotate each simulation point
        for r in spectrum:
            ax.annotate(f"ρ={r['rho']:.2f}\n{r['reduction_pct']:.0f}%",
                        (r["rho"], r["reduction_pct"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                        color="#2E86AB")

        ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Queue utilisation ρ", fontsize=12)
        ax.set_ylabel("Short-request P50 reduction vs. FCFS (%)", fontsize=11)
        ax.set_title("SJF benefit scales with queue pressure", fontsize=13)
        ax.set_xlim(0.2, 1.25)
        ax.set_ylim(0, 85)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.25)

        # Data source note
        fig.text(0.01, 0.01,
                 "Simulation calibrated to RTX 4090 Gemma3:4b: μ_short=3.5s, μ_long=8.9s, τ=3×μ_short=10.5s.",
                 fontsize=7, color="gray")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = out_dir / "workload_spectrum.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved {out_path}")

        # Also save JSON for paper records
        json_path = out_dir / "workload_spectrum.json"
        with open(json_path, "w") as f:
            json.dump({
                "simulation": spectrum,
                "burst": {"rho_approx": ">>1", "reduction_pct": burst_reduction,
                          "source": "RTX4090 Gemma3:4b benchmark n=250"},
            }, f, indent=2)
        print(f"Saved {json_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


# ── τ sensitivity plot (original) ─────────────────────────────────────────────

def plot_tau_sensitivity(results: list, fcfs: dict, out_dir: pathlib.Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tau_labels = [f"{r['tau_mult']}×" if r['tau_mult'] != float("inf") else "∞"
                      for r in results]
        short_p50 = [r["short_p50"] for r in results]
        short_p95 = [r["short_p95"] for r in results]
        long_p50  = [r["long_p50"]  for r in results]
        long_p95  = [r["long_p95"]  for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(tau_labels, short_p50, "o-", color="#2196F3", label="SHORT P50")
        ax1.plot(tau_labels, short_p95, "s--", color="#2196F3", alpha=0.6, label="SHORT P95")
        ax1.axhline(fcfs["short_p50"], color="red", linestyle=":", linewidth=1.5, label="FCFS SHORT P50")
        ax1.axhline(fcfs["short_p95"], color="red", linestyle="--", linewidth=1, alpha=0.6, label="FCFS SHORT P95")
        ax1.set_xlabel("Starvation timeout τ (× μ_short)", fontsize=11)
        ax1.set_ylabel("Sojourn time (s)", fontsize=11)
        ax1.set_title("Short-Request Latency vs. τ", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(short_p50, long_p50, "o-", color="#4CAF50", zorder=5)
        for i, label in enumerate(tau_labels):
            ax2.annotate(f"τ={label}", (short_p50[i], long_p50[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax2.plot(fcfs["short_p50"], fcfs["long_p50"], "r*", markersize=14,
                 label=f"FCFS (short={fcfs['short_p50']:.1f}s, long={fcfs['long_p50']:.1f}s)",
                 zorder=6)
        ax2.set_xlabel("Short P50 sojourn time (s)", fontsize=11)
        ax2.set_ylabel("Long P50 sojourn time (s)", fontsize=11)
        ax2.set_title("Pareto Frontier: Short vs. Long Latency", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / "tau_sensitivity.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)

    # 1. τ sensitivity at ρ=0.74
    tau_results, fcfs = tau_sensitivity_main()
    plot_tau_sensitivity(tau_results, fcfs, out_dir)

    # 2. Workload spectrum sweep
    spectrum = rho_sweep_main()
    plot_workload_spectrum(spectrum, out_dir)


if __name__ == "__main__":
    main()
