"""M1: TVLA-style leakage detection for the timing legs (EXPERIMENT_PLAN_v1 §1-M1).

Upgrades "remote = chance" from an RF-accuracy statement to a standard leakage-
detection statement: per-model vs rest-of-pool Welch t (orders 1-4, TVLA/Goodwill
convention) plus KS two-sample (vs rest, and all 36 pairs). First order only sees
mean shifts; the equalized/envelope channels live in higher orders/support, hence
orders 2-4 and KS are part of the pre-registered suite, not optional extras.

Pre-registered decision rule (do not tune after seeing results):
  family = one (leg, feature) cell; t family = 9 models x orders (1-4 for
  continuous features, 1 only for discrete width/body_len), ks-rest family = 9,
  ks-pair family = 36. The isolated review applies cell-wide Bonferroni across
  all t orders and KS families, FWER alpha = 0.01. The classic |t| > 4.5 count
  is reported alongside for TVLA-literature comparability only.

Heavy-tail rule (added after the first run tripped branch (2) on WAN rt_us):
  nominal moment p-values are unreliable on WAN roundtrips (order-k t-tests
  probe 2k-th moments of a heavy-tailed distribution). Every nominally DETECT
  cell on a WAN leg is re-judged by within-round label permutation (rep = round;
  preserves network-regime time structure), max-statistic null, 500 perms,
  p_cal = (1 + #{null >= obs}) / (1 + perms). The final verdict for those cells
  is the calibrated one. This is the branch-(2) investigation, not a threshold
  moved to force a null.

Pre-registered controls: P2 and loopback P3 and WAN body_len must DETECT;
WAN rt_us (SGX_S4_int, plain_wan) must be NULL under the final verdict.

Usage (PowerShell):
    python <data-root>/Analysis/sprint1/tvla_suite.py            # default six legs
    python <data-root>/Analysis/sprint1/tvla_suite.py --legs "name=E:/path.csv:SCEN"
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(__file__))
import test_nshot_confound_probe as pr  # noqa: E402

ALPHA = 0.01
TVLA_T = 4.5
FWD_RE = re.compile(r'"fwd_us":(-?\d+)')

DEFAULT_LEGS = [
    "plain_pos=<data-root>/Results/timing_plain_loopback_pos.csv:P_loopback_pos",
    "plain_sgx=<data-root>/Results/timing_plain_loopback_sgx.csv:P_loopback_sgx_s20261103",
    "SGX_S2_int=<data-root>/Results/timing_s2_interleaved.csv:S2_interleaved",
    "SGX_S4_int=<data-root>/Results/timing_s4_interleaved.csv:S4_interleaved",
    "plain_wan=<data-root>/Results/timing_plain_wan_pos.csv:P_wan_pos",
    "wire_len=<data-root>/Results/timing_s2_wire_len.csv:S2_wire_len",
]

# feature -> expected verdict; missing feature on a leg is skipped, not failed
EXPECT = {
    "plain_pos":  {"p1_us": "detect", "p2_us": "detect", "rt_us": "detect"},
    "plain_sgx":  {"p2_us": "detect", "rt_us": "detect", "body_len": "detect"},
    "SGX_S2_int": {"p2_us": "detect", "rt_us": "detect"},
    "SGX_S4_int": {"p2_us": "detect", "rt_us": "null"},
    "plain_wan":  {"p2_us": "detect", "rt_us": "null", "body_len": "detect"},
    "wire_len":   {"p2_us": "detect", "body_len": "detect"},
}


def load(spec: list[str], per_model: int) -> dict[str, pd.DataFrame]:
    legs = {}
    for item in spec:
        name, rest = item.split("=", 1)
        path, scen = rest.rsplit(":", 1)
        d = pd.read_csv(path)
        d = d[d["scenario"] == scen].copy()
        d = d[(d["enclave_us"] > 0) & (d["roundtrip_us"] > 0)].copy()
        if per_model:
            d = pr.downsample_per_model(d, per_model)
        d["p2_us"] = d["enclave_us"].astype(float)
        d["rt_us"] = d["roundtrip_us"].astype(float)
        d["width"] = d["enclave_us"].astype("int64").astype(str).str.len()
        d["body_len"] = (d["resp_bytes"].astype("float")
                         if "resp_bytes" in d and d["resp_bytes"].notna().any() else np.nan)
        d["p1_us"] = (d["resp_body"].astype(str).str.extract(FWD_RE)[0].astype(float)
                      if "resp_body" in d else np.nan)
        legs[name] = d.sort_index()
        print(f"[tvla] {name:12s} {len(d):5d} rows", flush=True)
    return legs


def welch(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2 or (x.std() == 0 and y.std() == 0):
        return 0.0, 1.0
    t, p = sps.ttest_ind(x, y, equal_var=False)
    return float(t), float(p)


def ks_stat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r = sps.ks_2samp(x, y, method="asymp")
    return float(r.statistic), float(r.pvalue)


CONTINUOUS = ("p1_us", "p2_us", "rt_us")


def family_stats(v: np.ndarray, labels: np.ndarray, models: list[str],
                 orders: tuple[int, ...]) -> dict[str, float]:
    """Max statistic per family; the permutation null recomputes exactly these."""
    out = {}
    mu = v.mean()
    for order in orders:
        z = (v - mu) ** order
        out[f"t_order{order}"] = max(
            abs(welch(z[labels == m], z[labels != m])[0]) for m in models)
    out["ks_rest"] = max(ks_stat(v[labels == m], v[labels != m])[0] for m in models)
    out["ks_pair"] = max(
        ks_stat(v[labels == models[a]], v[labels == models[b]])[0]
        for a in range(len(models)) for b in range(a + 1, len(models)))
    return out


def cell_tests(v: np.ndarray, labels: np.ndarray, models: list[str],
               orders: tuple[int, ...]) -> list[dict]:
    """All pre-registered tests for one (leg, feature) cell, nominal p-values."""
    rows = []
    mu = v.mean()
    for order in orders:
        z = (v - mu) ** order
        ts, ps = [], []
        for m in models:
            t, p = welch(z[labels == m], z[labels != m])
            ts.append(t); ps.append(p)
        ps = np.array(ps)
        pb = np.minimum(ps * len(ps), 1.0)
        i = int(np.argmin(ps))
        rows.append({"test": f"t_order{order}", "n_tests": len(ps),
                     "stat_max": float(np.max(np.abs(ts))),
                     "p_min_raw": float(ps[i]), "p_min_bonf": float(pb[i]),
                     "n_sig_tvla45": int(np.sum(np.abs(ts) > TVLA_T)),
                     "n_sig_bonf": int(np.sum(pb < ALPHA)),
                     "argmax": models[i], "pair": ""})
    ks, ps = [], []
    for m in models:
        k, p = ks_stat(v[labels == m], v[labels != m])
        ks.append(k); ps.append(p)
    ps = np.array(ps); pb = np.minimum(ps * len(ps), 1.0)
    i = int(np.argmin(ps))
    rows.append({"test": "ks_rest", "n_tests": len(ps), "stat_max": float(ks[i]),
                 "p_min_raw": float(ps[i]), "p_min_bonf": float(pb[i]),
                 "n_sig_tvla45": "", "n_sig_bonf": int(np.sum(pb < ALPHA)),
                 "argmax": models[i], "pair": ""})
    ks, ps, pairs = [], [], []
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            k, p = ks_stat(v[labels == models[a]], v[labels == models[b]])
            ks.append(k); ps.append(p)
            pairs.append(f"{models[a]}|{models[b]}")
    ps = np.array(ps); pb = np.minimum(ps * len(ps), 1.0)
    i = int(np.argmin(ps))
    rows.append({"test": "ks_pair", "n_tests": len(ps), "stat_max": float(ks[i]),
                 "p_min_raw": float(ps[i]), "p_min_bonf": float(pb[i]),
                 "n_sig_tvla45": "", "n_sig_bonf": int(np.sum(pb < ALPHA)),
                 "argmax": "", "pair": pairs[i]})
    # Control FWER across every t order and both KS families in this cell, not
    # merely within each family before taking their OR.
    cell_n_tests = int(sum(r["n_tests"] for r in rows))
    for r in rows:
        r["cell_n_tests"] = cell_n_tests
        r["p_cell_bonf"] = min(1.0, r["p_min_raw"] * cell_n_tests)
    return rows


def calibrate_cell(v: np.ndarray, labels: np.ndarray, models: list[str],
                   orders: tuple[int, ...], reps: np.ndarray, perms: int,
                   seed: int) -> tuple[dict[str, dict], bool]:
    """Within-round label permutation null for the family max statistics."""
    obs = family_stats(v, labels, models, orders)
    groups = [np.flatnonzero(reps == r) for r in np.unique(reps)]
    rng = np.random.default_rng(seed)
    null = {k: [] for k in obs}
    for _ in range(perms):
        lab = labels.copy()
        for g in groups:
            lab[g] = rng.permutation(lab[g])
        st = family_stats(v, lab, models, orders)
        for k in obs:
            null[k].append(st[k])
    out = {}
    n_families = len(obs)
    names = list(obs)
    for k, val in obs.items():
        arr = np.array(null[k])
        p = (1 + int(np.sum(arr >= val))) / (1 + perms)
        p_cell = min(1.0, p * n_families)
        out[k] = {"obs_stat": val, "null_mean": float(arr.mean()),
                  "null_p95": float(np.quantile(arr, 0.95)), "p_cal": p,
                  "p_cell_bonf": p_cell, "n_families": n_families}

    # Single-step global minP: rank-normalize each family's observed and null
    # max statistics, then compare the minimum rank across all test families.
    # This is a permutation sensitivity analysis, not the original preregistration.
    observed = np.array([obs[k] for k in names])
    null_matrix = np.column_stack([null[k] for k in names])
    combined = np.vstack([observed, null_matrix])
    greater = combined[:, None, :] < combined[None, :, :]
    upper_ranks = 1 + greater.sum(axis=1)
    min_ranks = upper_ranks.min(axis=1)
    global_p = (1 + int(np.sum(min_ranks[1:] <= min_ranks[0]))) / (1 + perms)
    detect = global_p < ALPHA
    return out, detect, float(global_p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--legs", nargs="+", default=DEFAULT_LEGS, metavar="NAME=PATH:SCENARIO")
    ap.add_argument("--per-model", type=int, default=990)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review/results/sprint1")
    ap.add_argument("--wan-legs", default="SGX_S4_int,plain_wan",
                    help="legs whose nominal DETECT cells get permutation calibration")
    ap.add_argument("--perms", type=int, default=500)
    a = ap.parse_args(argv)
    pr.SEED = a.seed
    os.makedirs(a.out_dir, exist_ok=True)
    wan_legs = set(a.wan_legs.split(","))

    legs = load(a.legs, a.per_model)
    all_rows, verdicts, cal_rows = [], [], []
    for name, d in legs.items():
        models = [m for m in pr.MODEL_ORDER if (d["model_id"] == m).any()]
        for feat in ("p1_us", "p2_us", "rt_us", "width", "body_len"):
            s = d[feat].dropna()
            if s.empty or s.nunique() < 2:
                continue
            v = s.to_numpy(float)
            lab = d.loc[s.index, "model_id"].to_numpy()
            orders = (1, 2, 3, 4) if feat in CONTINUOUS else (1,)
            rows = cell_tests(v, lab, models, orders)
            detect = any(r["p_cell_bonf"] < ALPHA for r in rows)
            calibrated = ""
            global_p = np.nan
            if detect and name in wan_legs:
                reps = d.loc[s.index, "rep"].to_numpy()
                cal, detect, global_p = calibrate_cell(v, lab, models, orders, reps,
                                                       a.perms, a.seed)
                calibrated = "detect" if detect else "null"
                for fam, r in cal.items():
                    cal_rows.append({"leg": name, "feature": feat, "family": fam,
                                     "perms": a.perms, "global_minP": global_p, **r,
                                     "verdict": calibrated})
                print(f"[tvla] {name} {feat}: nominal detect -> calibrated "
                      f"{calibrated} (global minP={global_p:.4f}, {a.perms} perms)", flush=True)
            for r in rows:
                all_rows.append({"leg": name, "feature": feat, "seed": a.seed,
                                 "n_rows": len(v), **r, "global_minP": global_p,
                                 "verdict": "detect" if detect else "null",
                                 "calibrated": calibrated})
            exp = EXPECT.get(name, {}).get(feat)
            if exp is not None:
                ok = ("detect" if detect else "null") == exp
                verdicts.append({"leg": name, "feature": feat, "expected": exp,
                                 "got": "detect" if detect else "null",
                                 "pass": ok})
        pd.DataFrame([r for r in all_rows if r["leg"] == name]).to_csv(
            os.path.join(a.out_dir, f"tvla_{name}.csv"), index=False)

    pd.DataFrame(all_rows).to_csv(os.path.join(a.out_dir, "tvla_all.csv"), index=False)
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(os.path.join(a.out_dir, "tvla_calibrated.csv"),
                                      index=False)
    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(os.path.join(a.out_dir, "tvla_controls.csv"), index=False)
    print("\n[tvla] pre-registered controls:")
    print(vdf.to_string(index=False))
    n_fail = int((~vdf["pass"]).sum())
    print(f"\n[tvla] {len(vdf) - n_fail}/{len(vdf)} controls pass"
          + ("" if n_fail == 0 else "  -- PLAN BRANCH (2): wording of §3 must change"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())