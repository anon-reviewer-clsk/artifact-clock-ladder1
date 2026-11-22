"""RemoteID v3: isolated, finite-grid sequential identification.

The theorem and the implementation use separate round-disjoint data for
threshold selection and calibration intervals. All source CSVs are read-only;
all outputs are written below the isolated review workspace.

The finite-grid guarantee still requires independent requests and calibration
stationarity. Round bootstrap preserves round structure for resampling but does
not turn requests into independent experimental units or create new sessions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review")
R = Path(r"<data-root>/Results")
OUT = ROOT / "results/framework"
OUT.mkdir(parents=True, exist_ok=True)

LEGS = [
    ("cal_loopback", R / "timing_s2_interleaved.csv", "S2_interleaved"),
    ("cal_quant", R / "timing_s3_interleaved.csv", "S3_interleaved"),
    ("cal_wan", R / "timing_s4_interleaved.csv", "S4_interleaved"),
    ("cal_wan_rep", R / "timing_s4_interleaved_rep.csv", "S4_interleaved_rep"),
    ("real_loopback", R / "models_real/legs/timing_r2_loopback_sgx_s20261110.csv",
     "R2_loopback_sgx_s20261110"),
    ("real_wan", R / "models_real/legs/timing_r3_wan_s20261111.csv", "R3_wan_s20261111"),
    ("real_wan_rep", R / "models_real/legs/timing_wan_real_rep_s20261112.csv",
     "R3rep_sgx_s20261112"),
    ("tdx_wan", R / "models_real/legs/timing_t1_tdx_wan_s20261111.csv", "T1_tdx_sgx_s20261111"),
    ("tdx_loopback", R / "models_real/legs/timing_t2_tdx_loopback_s20261111.csv",
     "T2_tdx_loopback_sgx_s20261111"),
]
ALPHA = 0.05
NSCALES = 7
GRID = [2 ** k for k in range(5, 14)]
NBOOT = 40
BOOT_TARGET = GRID[-1]


def load(path: Path | str, scen: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[(d.scenario == scen) & (d.enclave_us > 0) & (d.roundtrip_us > 0)]
    return d.reset_index(drop=True)


def split_rounds(d: pd.DataFrame, frac: float = 0.6, seed: int = 7):
    rounds = np.random.default_rng(seed).permutation(np.sort(d["rep"].unique()))
    if len(rounds) < 3:
        raise ValueError("at least three rounds are required for calibration and test")
    ncal = min(len(rounds) - 1, max(2, int(len(rounds) * frac)))
    cal = d[d["rep"].isin(rounds[:ncal])]
    test = d[d["rep"].isin(rounds[ncal:])]
    if cal.empty or test.empty:
        raise ValueError("round split produced an empty partition")
    return cal.copy(), test.copy()


def split_calibration(cal: pd.DataFrame, frac: float = 0.5, seed: int = 17):
    """Return round-disjoint threshold-selection and CI-calibration data."""
    threshold_data, ci_data = split_rounds(cal, frac=frac, seed=seed)
    return threshold_data, ci_data


def make_X(rt: np.ndarray, spec: list[tuple[str, float]]) -> np.ndarray:
    rt = np.asarray(rt, dtype=float)
    return np.column_stack([
        (rt < s).astype(float) if k == "ind" else np.minimum(rt / s, 1.0)
        for k, s in spec
    ])


def mp_width(vhat: float, n: int, delta: float) -> float:
    """Two-sided fixed-time Maurer-Pontil half-width for a [0,1] mean."""
    if n < 2:
        return 1.0
    lg = np.log(2.0 / delta)
    return float(np.sqrt(2.0 * vhat * lg / n) + 7.0 * lg / (3.0 * (n - 1)))


def calibrate(cal: pd.DataFrame, models: list[str], delta_cal: float,
              threshold_data: pd.DataFrame):
    """Build fixed statistics on threshold_data and CIs on disjoint cal."""
    if set(threshold_data["rep"]).intersection(set(cal["rep"])):
        raise ValueError("threshold selection and CI calibration rounds overlap")
    meds = threshold_data.groupby("model_id")["roundtrip_us"].median()
    if set(models) != set(meds.index):
        raise ValueError("threshold data do not cover every candidate model")
    m_min, m_max = float(meds.min()), float(meds.max())
    spec = [("ind", m_min * f) for f in (0.5, 0.8, 1.1)]
    spec += [("clip", float(g)) for g in np.geomspace(m_min, m_max * 1.5, 4)]
    cis = {}
    for m in models:
        rt = cal.loc[cal.model_id == m, "roundtrip_us"].to_numpy(dtype=float)
        if len(rt) < 2:
            raise ValueError(f"insufficient CI calibration data for {m}")
        per = []
        for kind, sc in spec:
            x = (rt < sc).astype(float) if kind == "ind" else np.minimum(rt / sc, 1.0)
            p = float(x.mean())
            v = float(x.var(ddof=1))
            w = mp_width(v, len(x), delta_cal)
            per.append((p, max(0.0, p - w), min(1.0, p + w)))
        cis[m] = per
    return spec, cis


def stream_cis(X: np.ndarray, delta_grid: float):
    out = []
    for t in GRID:
        if t > len(X):
            break
        row = []
        for b in range(X.shape[1]):
            x = X[:t, b]
            p = float(x.mean())
            v = float(x.var(ddof=1))
            w = mp_width(v, t, delta_grid)
            row.append((p, max(0.0, p - w), min(1.0, p + w)))
        out.append((t, row))
    return out


def decide(cis_stream, cis_cal, models: list[str]):
    for t, sc in cis_stream:
        for candidate in models:
            separated = True
            for other in models:
                if other == candidate:
                    continue
                if not any(
                    sc[b][2] < cis_cal[other][b][1]
                    or sc[b][1] > cis_cal[other][b][2]
                    for b in range(len(sc))
                ):
                    separated = False
                    break
            if separated and all(
                sc[b][1] <= cis_cal[candidate][b][2]
                and sc[b][2] >= cis_cal[candidate][b][1]
                for b in range(len(sc))
            ):
                return candidate, t
    return None, None


def boot_rounds(test: pd.DataFrame, rng: np.random.Generator,
                target: int = BOOT_TARGET) -> np.ndarray:
    groups = [g.to_numpy(dtype=float) for _, g in test.groupby("rep")["roundtrip_us"]]
    if not groups:
        raise ValueError("cannot bootstrap an empty test set")
    parts, n = [], 0
    while n < target:
        g = groups[rng.integers(len(groups))]
        parts.append(g)
        n += len(g)
    return np.concatenate(parts)


def evaluate_streams(test: pd.DataFrame, models: list[str], spec, cis_cal,
                     delta_grid: float, rng: np.random.Generator, nboot: int):
    """Return one aligned record per original/bootstrap stream."""
    records = []
    for m_true in models:
        sub = test[test.model_id == m_true]
        for b in range(nboot):
            rt = sub.roundtrip_us.to_numpy(dtype=float) if b == 0 else boot_rounds(sub, rng)
            cs = stream_cis(make_X(rt, spec), delta_grid)
            m_hat, t = decide(cs, cis_cal, models)
            declared = m_hat is not None
            records.append({
                "true_model": m_true, "bootstrap_index": b,
                "declared": declared, "predicted_model": m_hat,
                "correct": bool(declared and m_hat == m_true),
                "declaration_t": t if declared else np.nan,
                "capped_asn": t if declared else GRID[-1],
                "observed_max_t": cs[-1][0] if cs else 0,
            })
    return records


def summarize_runs(records: list[dict]) -> dict:
    declared = [r for r in records if r["declared"]]
    errors = [not r["correct"] for r in declared]
    return {
        "runs": len(records), "n_decl": len(declared),
        "err_rate": float(np.mean(errors)) if errors else np.nan,
        "asn_med": float(np.median([r["capped_asn"] for r in records])),
        "asn_dec_med": float(np.median([r["declaration_t"] for r in declared]))
        if declared else np.nan,
        "trunc_frac": 1.0 - len(declared) / len(records) if records else np.nan,
    }


def single_shot_acc(cal: pd.DataFrame, test: pd.DataFrame, models: list[str]) -> float:
    meds = {m: np.log(cal.loc[cal.model_id == m, "roundtrip_us"]).median()
            for m in models}
    labels = test["model_id"].to_numpy()
    log_rt = np.log(test["roundtrip_us"].to_numpy(dtype=float))
    pred = np.array([min(models, key=lambda m: abs(log_rt[i] - meds[m]))
                     for i in range(len(log_rt))])
    return float((pred == labels).mean())


def gap_sigma(d: pd.DataFrame) -> float:
    g = d.groupby("model_id")["roundtrip_us"]
    meds, sds = g.median(), g.std()
    return float((meds.max() - meds.min()) / np.sqrt((sds ** 2).mean()))


def pairwise_ci_gaps(cis_cal, models: list[str], bins: int):
    """Positive gaps exactly correspond to disjoint calibration intervals."""
    rows = []
    for m in models:
        for other in models:
            if m == other:
                continue
            gap = max(
                max(
                    cis_cal[other][b][1] - cis_cal[m][b][2],
                    cis_cal[m][b][1] - cis_cal[other][b][2],
                )
                for b in range(bins)
            )
            rows.append({"model": m, "other": other, "gap": float(gap)})
    return rows


def predicted_full_declaration_budget(cis_cal, models: list[str], bins: int,
                                      delta_grid: float, vbar: float = 0.15) -> float:
    """Grid budget for every ordered pair, using the full MP width.

    This remains an interpretive prediction under the disclosed constant
    variance proxy, not a measured ASN or a theorem about identification power.
    """
    pair_budgets = []
    for m in models:
        for other in models:
            if m == other:
                continue
            bin_budgets = []
            for b in range(bins):
                distance = abs(cis_cal[m][b][0] - cis_cal[other][b][0])
                cal_width = cis_cal[other][b][2] - cis_cal[other][b][0]
                budget = np.inf
                for t in GRID:
                    if distance > cal_width + mp_width(vbar, t, delta_grid):
                        budget = t
                        break
                bin_budgets.append(budget)
            pair_budgets.append(min(bin_budgets))
    return float(max(pair_budgets)) if pair_budgets else np.nan


def main(nboot: int = NBOOT) -> None:
    rows, crit, all_runs = [], [], []
    for leg_index, (name, path, scen) in enumerate(LEGS):
        d = load(path, scen)
        models = sorted(d["model_id"].unique())
        K, B = len(models), NSCALES
        cal, test = split_rounds(d)
        threshold_data, ci_data = split_calibration(cal)
        dcal = ALPHA / (2 * B * K)
        dgrid = ALPHA / (2 * B * len(GRID))
        spec, cis_cal = calibrate(ci_data, models, dcal, threshold_data)
        rng = np.random.default_rng(7000 + leg_index)
        runs = evaluate_streams(test, models, spec, cis_cal, dgrid, rng, nboot)
        for r in runs:
            r["leg"] = name
        all_runs.extend(runs)
        summary = summarize_runs(runs)
        gaps = pairwise_ci_gaps(cis_cal, models, B)
        n_res = sum(g["gap"] > 0 for g in gaps)
        n_pred = predicted_full_declaration_budget(cis_cal, models, B, dgrid)
        a_obs = single_shot_acc(cal, test, models)
        r_value = gap_sigma(d)
        rows.append(dict(
            leg=name, K=K, n_threshold=len(threshold_data), n_calibration=len(ci_data),
            n_test=len(test), err_rate=summary["err_rate"],
            asn_med=summary["asn_med"], asn_dec_med=summary["asn_dec_med"],
            trunc_frac=summary["trunc_frac"],
            resolvable_pairs=f"{n_res}/{len(gaps)}", alpha=ALPHA,
        ))
        crit.append(dict(
            leg=name, chance=1.0 / K, r=r_value, a_obs=a_obs,
            asn_med=summary["asn_med"], n_pred=n_pred,
            gbot=min(g["gap"] for g in gaps),
        ))
        print(f"[fw] {name:14s} decl={summary['n_decl']}/{summary['runs']} "
              f"err={summary['err_rate']:.3f} ASN_cap={summary['asn_med']:.0f} "
              f"pairs={n_res}/{len(gaps)} a_obs={a_obs:.3f} r={r_value:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "sequential_error_control.csv", index=False)
    pd.DataFrame(all_runs).to_csv(OUT / "remoteid_runs.csv", index=False)
    cdf = pd.DataFrame(crit)
    preds = []
    for i in range(len(cdf)):
        tr = cdf.drop(index=i)
        ho = cdf.iloc[i]
        best = None
        for theta in np.geomspace(0.05, 50, 120):
            ft = lambda rr, ch: ch + (1 - ch) * (1 - np.exp(-rr / theta))
            err = np.mean((ft(tr.r.to_numpy(), tr.chance.to_numpy()) - tr.a_obs.to_numpy()) ** 2)
            if best is None or err < best[0]:
                best = (err, theta)
        theta = best[1]
        a_hat = ho.chance + (1 - ho.chance) * (1 - np.exp(-ho.r / theta))
        preds.append(dict(leg=ho.leg, theta=theta, a_obs=ho.a_obs,
                          a_pred=a_hat, abs_err=abs(a_hat - ho.a_obs)))
    pdf = pd.DataFrame(preds)
    pdf.to_csv(OUT / "criterion_prospective.csv", index=False)
    cdf.to_csv(OUT / "criterion_features.csv", index=False)
    print("\n[fw] leave-one-leg-out criterion:")
    print(pdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[fw] mean |pred-obs| = {pdf.abs_err.mean():.6f}")
    print(f"[fw] wrote {OUT}")


if __name__ == "__main__":
    main()
