"""M4: SNR sampling-complexity layer (EXPERIMENT_PLAN_v1 §1-M4).

Replaces the ad-hoc per-request gap/sigma criterion (the R6/R18/U8 family of
errors) with a detection-theoretic sample size, and -- because WAN RTT is
heavy-tailed -- validates the Gaussian formula against an empirical bootstrap
before trusting it.

Formula (one-sample mean test, known noise, small-SNR Gaussian limit of the
T-test complexity in des Noes, "Distribution of Signal to Noise Ratio and
Application to Leakage Detection", IACR TCHES 2024(2):384-402,
doi:10.46586/tches.v2024.i2.384-402):

    n* = ((z_{1-alpha} + z_{1-beta}) / d')^2,   d' = Delta / sigma  (per request)

For a proportion (bounded indicator) sigma = sqrt(p(1-p)). For a two-sample
(profiling-free) test multiply by 2. Decoy attenuation of the mean channel:
Delta_eff = (1-p) Delta_0  =>  n*(p) = n*(0) / (1-p)^2.

Part B (the load-bearing part): bootstrap n* on the empirical distributions.
Draw windows of size n from the leg's per-model rows, compute the statistic
under H0 (within-round permuted labels) and H1 (true labels), and find the
smallest n with power >= 1-beta at FP <= alpha. Gaussian prediction vs
bootstrap n* tells us where the formula is usable:
  - raw WAN rt mean: heavy tail -> formula expected to FAIL (moments unstable)
  - uf indicator (rt < session 11th pct): bounded -> formula expected to hold

Part C: back-predictions of the three closed thresholds, as a comparison table
(formula vs observed), honest verdict per row. No new collection.

Usage (PowerShell):
    python <data-root>/Analysis/sprint1/snr_complexity.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(__file__))
import test_nshot_confound_probe as pr  # noqa: E402
import tvla_suite as tv  # noqa: E402

ALPHA = 0.05
BETA = 0.05
# Match the two-sided bootstrap rejection rule used below.
Z_SUM = sps.norm.ppf(1 - ALPHA / 2) + sps.norm.ppf(1 - BETA)
N_GRID = (10, 25, 50, 100, 200, 500, 1000, 2000, 5000)
BOOT = 2000
SEED = 42


def n_star(dprime: float, two_sample: bool = False) -> float:
    if dprime <= 0:
        return float("inf")
    return (2.0 if two_sample else 1.0) * (Z_SUM / dprime) ** 2


def binary_dprime(x: np.ndarray, y: np.ndarray, indicator: bool) -> tuple[float, dict]:
    """Best-pair d' for the mean test (or proportion test if indicator)."""
    if indicator:
        px, py = x.mean(), y.mean()
        pbar = (px + py) / 2
        sigma = np.sqrt(pbar * (1 - pbar))
        dp = abs(px - py) / sigma if sigma > 0 else 0.0
        return float(dp), {"px": float(px), "py": float(py), "sigma": float(sigma)}
    mx, my = x.mean(), y.mean()
    sigma = np.sqrt((x.var() + y.var()) / 2)
    dp = abs(mx - my) / sigma if sigma > 0 else 0.0
    return float(dp), {"mx": float(mx), "my": float(my), "sigma": float(sigma)}


def bootstrap_nstar(v: np.ndarray, lab: np.ndarray, reps: np.ndarray,
                    mA: str, mB: str, indicator: bool,
                    threshold: float | None) -> dict:
    """Empirical n* for the best-pair mean/proportion test, within-round H0."""
    x = v[lab == mA]
    y = v[lab == mB]
    if indicator:
        x = (x < threshold).astype(float)
        y = (y < threshold).astype(float)
    dp, parts = binary_dprime(x, y, indicator)
    pred = n_star(dp, two_sample=True)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(reps == r) for r in np.unique(reps)]
    # precompute 200 within-round permuted null pools (exact round structure,
    # reused across boots and n -- the null is the same distribution for all n)
    vv = (v < threshold).astype(float) if indicator else v
    null_x, null_y = [], []
    for _ in range(200):
        lp = lab.copy()
        for g in groups:
            lp[g] = rng.permutation(lp[g])
        null_x.append(vv[lp == mA])
        null_y.append(vv[lp == mB])
    rows = []
    for n in N_GRID:
        h1 = np.empty(BOOT)
        h0 = np.empty(BOOT)
        for b in range(BOOT):
            h1[b] = rng.choice(x, size=n, replace=True).mean() - \
                rng.choice(y, size=n, replace=True).mean()
            k = b % 200
            h0[b] = rng.choice(null_x[k], size=n, replace=True).mean() - \
                rng.choice(null_y[k], size=n, replace=True).mean()
        cut = np.quantile(np.abs(h0), 1 - ALPHA)
        power = float(np.mean(np.abs(h1) > cut))
        rows.append({"n": n, "power": power, "cut": float(cut),
                     "h0_sd": float(h0.std()), "h1_mean": float(h1.mean())})
        if power >= 1 - BETA:
            break
    emp = rows[-1]["n"] if rows[-1]["power"] >= 1 - BETA else float("inf")
    return {"pair": f"{mA}|{mB}", "dprime": dp, **parts,
            "n_star_gauss": pred, "n_star_boot": emp,
            "boot_last_power": rows[-1]["power"], "trace": rows}


def best_pair(v: np.ndarray, lab: np.ndarray, models: list[str]) -> tuple[str, str]:
    best, bd = None, -1
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            D = tv.ks_stat(v[lab == models[a]], v[lab == models[b]])[0]
            if D > bd:
                bd, best = D, (models[a], models[b])
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review/results/sprint1")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    legs = tv.load([
        "SGX_S2_int=<data-root>/Results/timing_s2_interleaved.csv:S2_interleaved",
        "SGX_S4_int=<data-root>/Results/timing_s4_interleaved.csv:S4_interleaved",
    ], 990)
    rows = []
    for name, d in legs.items():
        models = [m for m in pr.MODEL_ORDER if (d["model_id"] == m).any()]
        v = d["rt_us"].to_numpy(float)
        lab = d["model_id"].to_numpy()
        reps = d["rep"].to_numpy()
        mA, mB = best_pair(v, lab, models)
        thr = float(np.quantile(v, 0.11))
        for indicator in (False, True):
            r = bootstrap_nstar(v, lab, reps, mA, mB, indicator, thr)
            trace = r.pop("trace")
            rows.append({"leg": name, "stat": "uf_proportion" if indicator else "raw_mean",
                         "alpha": ALPHA, "beta": BETA, **r})
            print(f"[m4] {name} {'uf' if indicator else 'mean':12s} pair={r['pair']} "
                  f"d'={r['dprime']:.4f}  n*_gauss={r['n_star_gauss']:.0f}  "
                  f"n*_boot={r['n_star_boot']}  (last power {r['boot_last_power']:.2f})",
                  flush=True)
            for t in trace:
                rows.append({"leg": name, "stat": ("uf_proportion" if indicator else "raw_mean")
                             + "_trace", "pair": r["pair"], **t})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, "snr_predicted_n.csv"), index=False)

    # --- Part C: back-predictions of the three closed thresholds ---
    # E8 decoy mean channel: d'_0 = 0.10/(1-0.1) = 0.111 (U8 §4 back-calc).
    d0 = 0.111
    print("\n[m4] Part C back-predictions (formula vs observed):")
    print(f"  E8 decoy mean channel d'_0={d0:.3f}:")
    for p in (0.0, 0.5, 0.75):
        ns = n_star((1 - p) * d0)
        print(f"    p={p:.2f}: n*_mean-channel = {ns:.0f}  "
              f"(observed learner: p=0 n=10 acc 0.414; p=0.5 n=100 acc 0.261; "
              f"p=0.75 n=100 acc 0.152)")
    print("    -> formula applied to the MEAN channel predicts no detection at any p "
          "for n<=100; the learner exploits the envelope/min channel, so the mean-channel "
          "formula is a LOWER bound on attacker power. Branch (2): empirical law only.")
    # E7 grid: sealing is a SUPPORT argument, not SNR: one bucket iff Delta >= support width.
    print("  E7 grid: sealing Delta>=3.4ms == emission support width (single-bucket "
          "condition). Support argument, SNR formula not applicable; consistent.")
    # WAN uf signature: the M1 finding gets a concrete detection budget.
    uf = out[(out.stat == "uf_proportion") & out.n_star_boot.notna()]
    print("\n[m4] WAN uf-signature detection budget (best pair, alpha=beta=0.05):")
    print(uf[["leg", "pair", "dprime", "n_star_gauss", "n_star_boot"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[m4] wrote {os.path.join(a.out_dir, 'snr_predicted_n.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())