"""Review statistics with shared held-out rounds and explicit estimands.

Wilson intervals are descriptive of correlated bootstrap replays, not CIs from
independent WAN sessions. RF and RemoteID use the same calibration/test rounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

import framework_core as fc
import test_nshot_confound_probe as pr

OUT = fc.OUT
BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def main(nboot: int = 20) -> None:
    rows, budget_rows, sigma_rows = [], [], []
    for leg_index, (name, path, scen) in enumerate(fc.LEGS):
        d = fc.load(path, scen)
        models = sorted(d.model_id.unique())
        K, B = len(models), fc.NSCALES
        cal, test = fc.split_rounds(d)
        threshold_data, ci_data = fc.split_calibration(cal)
        dcal = fc.ALPHA / (2 * B * K)
        dgrid = fc.ALPHA / (2 * B * len(fc.GRID))
        spec, cis_cal = fc.calibrate(ci_data, models, dcal, threshold_data)
        rng = np.random.default_rng(21000 + leg_index)
        runs = fc.evaluate_streams(test, models, spec, cis_cal, dgrid, rng, nboot)
        ndec = sum(r["declared"] for r in runs)
        ncor = sum(r["correct"] for r in runs)
        lo, hi = wilson(ncor, ndec)
        rows.append(dict(
            leg=name, runs=len(runs), n_decl=ndec, n_correct=ncor,
            err=1 - ncor / ndec if ndec else np.nan,
            err_ci_lo=1 - hi, err_ci_hi=1 - lo,
            ci_interpretation="descriptive_bootstrap_not_independent_sessions",
            abstain=1 - ndec / len(runs),
        ))
        for budget in BUDGETS:
            selected = [r for r in runs if r["declared"] and r["declaration_t"] <= budget]
            dec = len(selected)
            cor = sum(r["correct"] for r in selected)
            budget_rows.append(dict(
                leg=name, budget=budget, method="remoteid",
                decl_rate=dec / len(runs), correct_rate=cor / len(runs),
                abstain_rate=1 - dec / len(runs),
                err_given_decl=1 - cor / dec if dec else np.nan,
                evaluation="shared_round_holdout",
            ))
        if name in ("real_wan", "real_wan_rep", "tdx_wan", "cal_loopback"):
            for n in (32, 64, 128, 256):
                acc, _, _ = pr.rf_train_test(cal, test, models, n)
                if np.isnan(acc):
                    continue
                budget_rows.append(dict(
                    leg=name, budget=n, method="rf_window",
                    decl_rate=1.0, correct_rate=acc, abstain_rate=0.0,
                    err_given_decl=1 - acc, evaluation="shared_round_holdout",
                ))

        g = d.groupby("model_id")["roundtrip_us"]
        meds = g.median()
        span = meds.max() - meds.min()
        sig_sd = np.sqrt((g.std() ** 2).mean())
        sig_iqr = np.sqrt((((g.quantile(0.75) - g.quantile(0.25)) / 1.349) ** 2).mean())
        sig_mad = np.sqrt((g.apply(lambda x: 1.4826 * np.median(np.abs(x - x.median()))) ** 2).mean())
        mv = meds.to_numpy()
        pair_gaps = [abs(mv[i] - mv[j]) for i in range(len(mv)) for j in range(i + 1, len(mv))]
        a_obs = fc.single_shot_acc(cal, test, models)
        sigma_rows.append(dict(
            leg=name, chance=1 / K, a_obs=a_obs,
            r_sd=span / sig_sd, r_iqr=span / sig_iqr,
            r_mad=span / sig_mad, r_pair=float(np.mean(pair_gaps)) / sig_sd,
        ))
        print(f"[rv] {name:14s} decl={ndec}/{len(runs)} abstain={1-ndec/len(runs):.2f} "
              f"r_sd={span/sig_sd:.2f} a_obs={a_obs:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "declaration_counts.csv", index=False)
    bdf = pd.DataFrame(budget_rows)
    bdf.to_csv(OUT / "same_budget.csv", index=False)
    sdf = pd.DataFrame(sigma_rows)
    sdf.to_csv(OUT / "sigma_sensitivity.csv", index=False)
    print("\n[rv] Spearman with a_obs across legs (mixed chance levels; descriptive):")
    for c in ["r_sd", "r_iqr", "r_mad", "r_pair"]:
        rho, p = sps.spearmanr(sdf[c], sdf.a_obs)
        print(f"  {c:7s} rho={rho:+.3f} p={p:.4f}")
    print(f"[rv] wrote {OUT}")


if __name__ == "__main__":
    main()
