"""Bidirectional R3/R3rep evaluation on aligned request keys.

Primary protocol: profile on one date and test on the other. The unseen-sample
variant additionally uses sample_id 0-49 for training and 50-99 for testing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import framework_core as fc
import test_nshot_confound_probe as pr

OUT = fc.ROOT / "results/cross_session"
OUT.mkdir(parents=True, exist_ok=True)
A_PATH = Path(r"<data-root>/Results/models_real/legs/timing_r3_wan_s20261111.csv")
B_PATH = Path(r"<data-root>/Results/models_real/legs/timing_wan_real_rep_s20261112.csv")
A_SCEN = "R3_wan_s20261111"
B_SCEN = "R3rep_sgx_s20261112"
KEYS = ["model_id", "rep", "sample_id"]
BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192]
N_WINDOWS = [1, 10, 32, 50, 64, 100, 128, 256]


def aligned_sessions(mode: str = "round_budget"):
    """Align sessions without inventing paired samples.

    round_budget uses min(count) within each model×round, retaining collection
    order; different seeds may choose different sample_ids. common_keys is a
    smaller sensitivity subset with exact model×round×sample_id matches.
    """
    a = fc.load(A_PATH, A_SCEN)
    b = fc.load(B_PATH, B_SCEN)
    a["__order"] = np.arange(len(a))
    b["__order"] = np.arange(len(b))
    if mode == "common_keys":
        common = a[KEYS].merge(b[KEYS], on=KEYS, how="inner")
        a = a.merge(common, on=KEYS, how="inner").sort_values("__order").reset_index(drop=True)
        b = b.merge(common, on=KEYS, how="inner").sort_values("__order").reset_index(drop=True)
    elif mode == "round_budget":
        group_keys = ["model_id", "rep"]
        counts_a = a.groupby(group_keys).size().rename("n_a")
        counts_b = b.groupby(group_keys).size().rename("n_b")
        groups = counts_a.to_frame().join(counts_b.to_frame(), how="inner")
        groups["n"] = groups[["n_a", "n_b"]].min(axis=1)
        keep_a, keep_b = [], []
        for (model, rep), row in groups.iterrows():
            n = int(row.n)
            ga = a[(a.model_id == model) & (a.rep == rep)].sort_values("__order").head(n)
            gb = b[(b.model_id == model) & (b.rep == rep)].sort_values("__order").head(n)
            keep_a.append(ga)
            keep_b.append(gb)
        a = pd.concat(keep_a).sort_values("__order").reset_index(drop=True)
        b = pd.concat(keep_b).sort_values("__order").reset_index(drop=True)
    else:
        raise ValueError(mode)
    if a[KEYS].duplicated().any() or b[KEYS].duplicated().any():
        raise ValueError("duplicate request keys prevent aligned evaluation")
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("aligned sessions have inconsistent sizes")
    return a, b


def variant_frames(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    if variant in ("cross_session", "common_keys"):
        return train, test
    if variant == "cross_session_unseen_samples":
        return train[train.sample_id < 50].copy(), test[test.sample_id >= 50].copy()
    raise ValueError(variant)


def evaluate_direction(train: pd.DataFrame, test: pd.DataFrame, direction: str,
                       variant: str, models: list[str], nboot: int, seed: int):
    train, test = variant_frames(train, test, variant)
    rf_rows, per_model_rows = [], []
    for window_mode in ("sequential", "random"):
        for n in N_WINDOWS:
            acc, pred, y = pr.rf_train_test(train, test, models, n, window_mode=window_mode)
            if np.isnan(acc):
                continue
            rf_rows.append(dict(
                direction=direction, variant=variant, window_mode=window_mode,
                n=n, accuracy=acc,
                train_windows=sum(len(train[train.model_id == m]) // n for m in models),
                test_windows=sum(len(test[test.model_id == m]) // n for m in models),
            ))
            for i, model in enumerate(models):
                mask = y == i
                per_model_rows.append(dict(
                    direction=direction, variant=variant, window_mode=window_mode,
                    n=n, model=model, test_windows=int(mask.sum()),
                    accuracy=float((pred[mask] == i).mean()) if mask.any() else np.nan,
                ))

    threshold_data, ci_data = fc.split_calibration(train, frac=0.5, seed=17)
    K, B = len(models), fc.NSCALES
    dcal = fc.ALPHA / (2 * B * K)
    dgrid = fc.ALPHA / (2 * B * len(fc.GRID))
    spec, cis_cal = fc.calibrate(ci_data, models, dcal, threshold_data)
    rng = np.random.default_rng(seed)
    runs = fc.evaluate_streams(test, models, spec, cis_cal, dgrid, rng, nboot)
    for r in runs:
        r.update(direction=direction, variant=variant)
    budget_rows = []
    for budget in BUDGETS:
        selected = [r for r in runs if r["declared"] and r["declaration_t"] <= budget]
        dec = len(selected)
        cor = sum(r["correct"] for r in selected)
        budget_rows.append(dict(
            direction=direction, variant=variant, budget=budget,
            runs=len(runs), declarations=dec, correct=cor,
            decl_rate=dec / len(runs), correct_rate=cor / len(runs),
            abstain_rate=1 - dec / len(runs),
            err_given_decl=1 - cor / dec if dec else np.nan,
        ))
    nearest = fc.single_shot_acc(train, test, models)
    print(f"[xs] {direction:11s} {variant:30s} nearest={nearest:.3f} "
          f"RF(n=100)={next((r['accuracy'] for r in rf_rows if r['n'] == 100), np.nan):.3f} "
          f"RemoteID decl@8192={budget_rows[-1]['decl_rate']:.3f}", flush=True)
    return rf_rows, per_model_rows, runs, budget_rows, nearest


def main(nboot: int = 20) -> None:
    a, b = aligned_sessions(mode="round_budget")
    common_a, common_b = aligned_sessions(mode="common_keys")
    pd.DataFrame([
        {"alignment": "round_budget", "rows_per_session": len(a)},
        {"alignment": "common_keys_sensitivity", "rows_per_session": len(common_a)},
    ]).to_csv(OUT / "alignment_summary.csv", index=False)
    models = sorted(set(a.model_id) | set(b.model_id))
    counts = []
    for model in models:
        counts.append(dict(model=model, aligned_a=int((a.model_id == model).sum()),
                           aligned_b=int((b.model_id == model).sum())))
    pd.DataFrame(counts).to_csv(OUT / "aligned_counts.csv", index=False)

    rf_rows, per_model_rows, all_runs, budget_rows, nearest_rows = [], [], [], [], []
    for index, (train, test, direction) in enumerate(((a, b, "R3_to_R3rep"), (b, a, "R3rep_to_R3"))):
        for variant in ("cross_session", "cross_session_unseen_samples"):
            rf, pm, runs, budgets, nearest = evaluate_direction(
                train, test, direction, variant, models, nboot,
                seed=33000 + 10 * index + len(variant),
            )
            rf_rows.extend(rf)
            per_model_rows.extend(pm)
            all_runs.extend(runs)
            budget_rows.extend(budgets)
            nearest_rows.append(dict(direction=direction, variant=variant, accuracy=nearest))
    for index, (train, test, direction) in enumerate(((common_a, common_b, "R3_to_R3rep"),
                                                       (common_b, common_a, "R3rep_to_R3"))):
        rf, pm, runs, budgets, nearest = evaluate_direction(
            train, test, direction, "common_keys", models, nboot,
            seed=34000 + index,
        )
        rf_rows.extend(rf)
        per_model_rows.extend(pm)
        all_runs.extend(runs)
        budget_rows.extend(budgets)
        nearest_rows.append(dict(direction=direction, variant="common_keys", accuracy=nearest))

    pd.DataFrame(rf_rows).to_csv(OUT / "cross_session_rf.csv", index=False)
    pd.DataFrame(per_model_rows).to_csv(OUT / "cross_session_rf_per_model.csv", index=False)
    pd.DataFrame(all_runs).to_csv(OUT / "cross_session_remoteid_runs.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(OUT / "cross_session_remoteid_budgets.csv", index=False)
    pd.DataFrame(nearest_rows).to_csv(OUT / "cross_session_nearest_centroid.csv", index=False)
    print(f"[xs] aligned rows={len(a)} per session; wrote {OUT}")


if __name__ == "__main__":
    main()
