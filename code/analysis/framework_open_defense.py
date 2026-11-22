"""Open-set evaluation and explicitly non-adaptive defense stress tests.

Each open-set fold rebuilds statistics using known models only. Defense tests
transform only the test stream: they measure calibration-distribution mismatch,
not a defense-aware attacker and not the validity of Theorem 1 after defense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import framework_core as fc

OUT = fc.OUT


def checkpoint_power(open_rows: pd.DataFrame, checkpoint: int) -> float:
    selected = open_rows[open_rows.t == checkpoint]
    return float(selected.power.mean()) if not selected.empty else float("nan")


def open_set(d: pd.DataFrame, models: list[str], rng: np.random.Generator, nboot: int = 20):
    cal, test = fc.split_rounds(d)
    threshold_data, ci_data = fc.split_calibration(cal)
    power_rows, false_rows = [], []
    for unknown in models:
        known = [m for m in models if m != unknown]
        # The unknown model is excluded from thresholds, scales, and calibration CIs.
        known_threshold = threshold_data[threshold_data.model_id.isin(known)]
        known_ci = ci_data[ci_data.model_id.isin(known)]
        K, B = len(known), fc.NSCALES
        dcal = fc.ALPHA / (2 * B * K)
        dgrid = fc.ALPHA / (2 * B * len(fc.GRID))
        scales, cis_cal = fc.calibrate(known_ci, known, dcal, known_threshold)

        sub = test[test.model_id == unknown]
        rejection = {t: 0 for t in fc.GRID}
        for b in range(nboot):
            rt = sub.roundtrip_us.to_numpy(dtype=float) if b == 0 else fc.boot_rounds(sub, rng)
            cs = fc.stream_cis(fc.make_X(rt, scales), dgrid)
            observed = {t for t, _ in cs}
            for t, sc in cs:
                reject = all(any(
                    sc[b][2] < cis_cal[m][b][1] or sc[b][1] > cis_cal[m][b][2]
                    for b in range(B)
                ) for m in known)
                rejection[t] += int(reject)
            # Unobserved checkpoints remain zero: this stream did not reject there.
            for t in fc.GRID:
                if t not in observed:
                    rejection[t] += 0
        for t in fc.GRID:
            power_rows.append(dict(unknown=unknown, t=t, power=rejection[t] / nboot))

        # One actual stream per known model in this fold; a run counts at most once.
        for true_model in known:
            sub_known = test[test.model_id == true_model]
            cs = fc.stream_cis(fc.make_X(sub_known.roundtrip_us.to_numpy(dtype=float), scales), dgrid)
            rejected_at = []
            for t, sc in cs:
                reject = all(any(
                    sc[b][2] < cis_cal[m][b][1] or sc[b][1] > cis_cal[m][b][2]
                    for b in range(B)
                ) for m in known)
                if reject:
                    rejected_at.append(t)
            false_rows.append(dict(
                unknown_fold=unknown, true_model=true_model,
                false_rejected=bool(rejected_at),
                first_false_rejection_t=min(rejected_at) if rejected_at else np.nan,
            ))
    return pd.DataFrame(power_rows), pd.DataFrame(false_rows)


def defense_stress(d: pd.DataFrame, models: list[str], rng: np.random.Generator,
                   nboot: int = 10) -> pd.DataFrame:
    """Non-adaptive mismatch stress; no re-calibration and no adaptive retraining."""
    cal, test = fc.split_rounds(d)
    threshold_data, ci_data = fc.split_calibration(cal)
    K, B = len(models), fc.NSCALES
    dcal = fc.ALPHA / (2 * B * K)
    dgrid = fc.ALPHA / (2 * B * len(fc.GRID))
    scales, cis_cal = fc.calibrate(ci_data, models, dcal, threshold_data)
    med = float(threshold_data.roundtrip_us.median())
    pool = threshold_data.roundtrip_us.to_numpy(dtype=float)
    rows = []
    configs = [("none", 0, 0.0)]
    configs += [(f"quant_{q}x", q, 0.0) for q in (0.25, 0.5, 1, 2, 4)]
    configs += [(f"decoy_p{p}", 0, p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for name, q, p in configs:
        records = []
        for m_true in models:
            sub = test[test.model_id == m_true]
            for b in range(nboot):
                rt = (sub.roundtrip_us.to_numpy(dtype=float) if b == 0
                      else fc.boot_rounds(sub, rng)).copy()
                if q > 0:
                    delta = q * med
                    rt = np.ceil(rt / delta) * delta
                if p > 0:
                    mask = rng.random(len(rt)) < p
                    rt[mask] = rng.choice(pool, size=mask.sum())
                cs = fc.stream_cis(fc.make_X(rt, scales), dgrid)
                mh, t = fc.decide(cs, cis_cal, models)
                declared = mh is not None
                records.append({
                    "true_model": m_true, "declared": declared,
                    "correct": bool(declared and mh == m_true),
                    "declaration_t": t if declared else np.nan,
                    "capped_asn": t if declared else fc.GRID[-1],
                })
        summary = fc.summarize_runs(records)
        rows.append(dict(
            defense=name, evaluation_mode="nonadaptive_mismatch_only",
            err=summary["err_rate"], decl_rate=summary["n_decl"] / len(records),
            asn_med=summary["asn_med"], asn_dec_med=summary["asn_dec_med"],
            trunc_frac=summary["trunc_frac"],
        ))
    return pd.DataFrame(rows)


def main(open_nboot: int = 20, defense_nboot: int = 10) -> None:
    all_op, all_fr, all_def = [], [], []
    for leg_index, (name, path, scen) in enumerate(fc.LEGS):
        d = fc.load(path, scen)
        models = sorted(d.model_id.unique())
        rng = np.random.default_rng(11000 + leg_index)
        op, fr = open_set(d, models, rng, nboot=open_nboot)
        op["leg"] = name
        fr["leg"] = name
        all_op.append(op)
        all_fr.append(fr)
        power_2048 = checkpoint_power(op, 2048)
        power_8192 = checkpoint_power(op, 8192)
        print(f"[os] {name:14s} false_reject_runs={int(fr.false_rejected.sum())}/{len(fr)} "
              f"power@2048={power_2048:.2f} power@8192={power_8192:.2f}", flush=True)
        df = defense_stress(d, models, rng, nboot=defense_nboot)
        df["leg"] = name
        all_def.append(df)
    pd.concat(all_op).to_csv(OUT / "open_set_power.csv", index=False)
    pd.concat(all_fr).to_csv(OUT / "open_set_false_rejection.csv", index=False)
    pd.concat(all_def).to_csv(OUT / "defense_stress.csv", index=False)
    print(f"[fw2] wrote {OUT}")


if __name__ == "__main__":
    main()
